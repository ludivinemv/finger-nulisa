# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
# Script for Nulisa biomarkers analyses in total cohort (disregards of the group)
# NULISA NPQ used as is (already log2); Lumipulse pTau217 log-transformed
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////



# /////////////////////////////////////////////////////////////////////////////
# Arguments -------------------------------------------------------------------
# /////////////////////////////////////////////////////////////////////////////

## Options ----
version <- "2-0"

color_carrier ="#0072B2"
color_noncarrier= "#E69F00"
color_control="#4D4D4D"
color_interv = "#CC79A7"



# /////////////////////////////////////////////////////////////////////////////
# Commune script for all parts ------------------------------------------------
# /////////////////////////////////////////////////////////////////////////////

# .............................................................................
## Packages -------------------------------------------------------------------
# .............................................................................
## List of packages to install and load ----
packages <- c("dplyr", 
              "ggplot2", 
              "lme4", 
              "tidyr", 
              "readr",
              "lmerTest", 
              "broom.mixed",
              "broom",
              "purrr", 
              "stringr",
              "forcats",
              "patchwork",
              "ggrepel",
              "emmeans",
              "remotes",
              "colorspace",
              "svglite")




# Loop to install packages
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) { # Install only if the package is not already installed
    install.packages(pkg)
  }
}


# Loop to load packages
for (pkg in packages) {
  library(pkg, character.only = TRUE)
}



# ..............................................................................
## Variables definition  -------------------------------------------------------
# ..............................................................................

list_colors <- c(
  "carrier" = color_carrier,
  "noncarrier" = color_noncarrier,
  "interv" = color_interv,
  "control" = color_control
)

# Biomarkers excluded from ALL models
# - core AD biomarkers (already excluded from the figures)
# - NULISA APOE4: reflects the APOE e4 allele (genotype-driven distribution)
# Excluding them here also means they are not counted in the FDR correction.
# Lumipulse pTau217 is still used as the OUTCOME in the pTau217-outcome models and as a comparator in the correlation heatmaps.
EXCLUDE_FROM_MODELS <- c(
  "pTau181", "pTau231", "pTau217",
  "AB38", "AB40", "AB42",
  "NEFL", "GFAP",
  "APOE4",
  "lumipulse_ptau217"
)




# ..............................................................................
## Path definition -------------------------------------------------------------
# ..............................................................................
folder_path <- "../data/"
processed_path <- paste0(folder_path,"processed/")
output_path <- paste0(folder_path,"outputs_R/total_cohort/")
  
if (!dir.exists(output_path)) {dir.create(output_path, recursive = TRUE)}# Create the folder if it does not exist
for (fold in c("Longitudinal","cross-sectional","Stratified")){
  if (!dir.exists(paste0(output_path,"/",fold))) {dir.create(paste0(output_path,"/",fold))}# Create the folder if it does not exist
  
}




# ..............................................................................
## Functions -------------------------------------------------------------------
# ..............................................................................


source("functions/plots_and_tables")
source("functions/utilities")
source("functions/preprocessing")
source("functions/statistics")




# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
#  baseline biomarkers VS baseline cognition----------------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////

# Purpose:
# Cross-sectional associations between baseline biomarkers
# and baseline cognition in the total cohort
#
# Outcome:
#   ztotal_composite_BL2
#
# Predictors:
#   all baseline NULISA biomarkers (_0)
#   + lumipulse_ptau217_0
#
# Covariate adjustment:
#   age_bl1
#   sex
#   educ_years_BL1
#.  bmi_bl2
#   campid
#
# Sample:
#   total cohort
#
# Processing steps for each biomarker:
#   Keep participants with:
#      - non-missing baseline biomarker
#      - non-missing baseline cognition
#      - non-missing covariates
#   Winsorize biomarker values at the 1st and 99th percentiles
#   Z-scale 
#   Fit a linear regression model:
#        ztotal_composite_BL2 ~ biomarker_z +
#          age_bl1 + sex + educ_years_BL1 +bmi_bl2 + campid 
#   6. Extract biomarker beta, standard error, and p-value
#   7. Apply FDR correction across biomarkers
#
# Reference levels:
#   sex: 1 = man (reference)
#   campid: 101 = Helsinki (reference)
#
# Output:
#   One row per biomarker with:
#   - beta
#   - standard error
#   - raw p-value
#   - FDR-adjusted p-value
# /////////////////////////////////////////////////////////
# .............................................................................
# File path
# .............................................................................
file_path <- paste0(processed_path,"dataset_original_clean.csv")

# .............................................................................
# Load data
# .............................................................................
df <- read_csv(file_path, show_col_types = FALSE)
df_all <- df %>%
  mutate(
    mITT = if_else(
      !is.na(ztotal_composite_12kk) | !is.na(ztotal_composite_24kk),
      1,
      0
    )
  )


df_all <- df_all %>%
  filter(mITT == 1)
# .............................................................................
# Identify biomarkers
# .............................................................................
nulisa_0 <- names(df_all)[grepl("^nulisa_.*_0$", names(df_all))]
biomarkers <- c(nulisa_0, "lumipulse_ptau217_0")
biomarkers <- drop_excluded(biomarkers)

# .............................................................................
# Run one biomarker model
# .............................................................................
run_cross <- function(bm, df) {
  
  bm_use <- bm
  
  dat <- df %>%
    select(
      ID,
      ztotal_composite_BL2,
      age_bl1,
      sex,
      educ_years_BL1, 
      bmi_bl2,
      campid,
      all_of(bm_use)
    ) %>%
    filter(
      !is.na(.data[[bm_use]]),
      !is.na(ztotal_composite_BL2),
      !is.na(age_bl1),
      !is.na(sex),
      !is.na(bmi_bl2),
      !is.na(educ_years_BL1),
      !is.na(campid)
    ) %>%
    mutate(
      biomarker_w = winsorize(.data[[bm_use]], 0.01)
    )
  
  # log-transform requires positive values
  if ((needs_log(bm) && any(dat$biomarker_w <= 0, na.rm = TRUE))) {
    return(tibble(
      biomarker = str_remove(str_remove(bm, "^nulisa_"), "_0$"),
      n = nrow(dat),
      beta = NA_real_,
      se = NA_real_,
      p = NA_real_,
      error = "Non-positive values after winsorization; log not possible"
    ))
  }
  
  dat <- dat %>%
    mutate(
      biomarker_log = log_transformation(biomarker_w, bm),
      biomarker_z = zscore_safe(biomarker_log),
      sex = relevel(factor(sex), ref = "1"),
      campid = relevel(factor(campid), ref = "101")
    ) %>%
    filter(!is.na(biomarker_z))
  
  model <- tryCatch(
    lm(
      ztotal_composite_BL2 ~ biomarker_z +
        age_bl1 + sex + educ_years_BL1 + bmi_bl2 + campid,
      data = dat
    ),
    error = function(e) e
  )
  
  if (inherits(model, "error")) {
    return(tibble(
      biomarker = str_remove(str_remove(bm, "^nulisa_"), "_0$"),
      n = nrow(dat),
      beta = NA_real_,
      se = NA_real_,
      p = NA_real_,
      error = model$message
    ))
  }
  
  
  #out1 <- tidy(model) %>%
  #  filter(term == "biomarker_z")
  #
  #
  #tibble(
  #  biomarker = str_remove(str_remove(bm, "^nulisa_"), "_0$"),
  #  n = nrow(dat),
  #  beta = out$estimate,
  #  se = out$std.error,
  #  p = out$p.value,
  #  error = NA_character_
  
  #)
    
    fixed <- tidy(model) %>%
      select(term, estimate, std.error, p.value)
    
    tibble(
      biomarker = str_remove(str_remove(bm, "^nulisa_"), "_0$"),
      n = nrow(dat),
      
      `Intercept coeff` = get_term_value(fixed, "(Intercept)", estimate),
      `Intercept std` = get_term_value(fixed, "(Intercept)", std.error),
      `Intercept pvalue` = get_term_value(fixed, "(Intercept)", p.value),
      
      `biomarker coeff` = get_term_value(fixed, "biomarker_z", estimate),
      `biomarker std` = get_term_value(fixed, "biomarker_z", std.error),
      `biomarker pvalue` = get_term_value(fixed, "biomarker_z", p.value),
      
      `age_0y coeff` = get_term_value(fixed, "age_bl1", estimate),
      `age_0y std` = get_term_value(fixed, "age_bl1", std.error),
      `age_0y pvalue` = get_term_value(fixed, "age_bl1", p.value),
      
      `sex coeff` = get_term_value(fixed, "sex2", estimate),
      `sex std` = get_term_value(fixed, "sex2", std.error),
      `sex pvalue` = get_term_value(fixed, "sex2", p.value),
      
      `educ_years_0y coeff` = get_term_value(fixed, "educ_years_BL1", estimate),
      `educ_years_0y std` = get_term_value(fixed, "educ_years_BL1", std.error),
      `educ_years_0y pvalue` = get_term_value(fixed, "educ_years_BL1", p.value),
      
      `bmi_bl2 coeff` = get_term_value(fixed, "bmi_bl2", estimate),
      `bmi_bl2 std` = get_term_value(fixed, "bmi_bl2", std.error),
      `bmi_bl2 pvalue` = get_term_value(fixed, "bmi_bl2", p.value),
      
      
      `campid201 coeff` = get_term_value(fixed, "campid201", estimate),
      `campid201 std`   = get_term_value(fixed, "campid201", std.error),
      `campid201 pvalue`= get_term_value(fixed, "campid201", p.value),
      
      `campid301 coeff` = get_term_value(fixed, "campid301", estimate),
      `campid301 std`   = get_term_value(fixed, "campid301", std.error),
      `campid301 pvalue`= get_term_value(fixed, "campid301", p.value),
      
      `campid401 coeff` = get_term_value(fixed, "campid401", estimate),
      `campid401 std`   = get_term_value(fixed, "campid401", std.error),
      `campid401 pvalue`= get_term_value(fixed, "campid401", p.value),
      
      `campid501 coeff` = get_term_value(fixed, "campid501", estimate),
      `campid501 std`   = get_term_value(fixed, "campid501", std.error),
      `campid501 pvalue`= get_term_value(fixed, "campid501", p.value),
      
      `campid601 coeff` = get_term_value(fixed, "campid601", estimate),
      `campid601 std`   = get_term_value(fixed, "campid601", std.error),
      `campid601 pvalue`= get_term_value(fixed, "campid601", p.value),
      
      error = NA_character_
    )
}

# .............................................................................
# Run all biomarker models
# .............................................................................
results <- map_dfr(biomarkers, run_cross, df = df_all)

# .............................................................................
# FDR correction
# .............................................................................
#results <- results %>%
#  mutate(
#    p_FDR = p.adjust(p, method = "fdr")
#  ) %>%
#  arrange(p)

p_cols <- grep("pvalue$", names(results), value = TRUE)

for (pcol in p_cols) {
  fdr_col <- paste0(pcol, " FDR")
  results[[fdr_col]] <- p.adjust(results[[pcol]], method = "fdr")
}
# .............................................................................
# Save results
# .............................................................................
write_csv(
  results,
  paste0(output_path,"cross-sectional/cross_sectional_baseline_biomarkers_vs_total_cognition_imputed_bmi_adj_mitt.csv")
)

print(results)

# .............................................................................
# Top bm 
# .............................................................................

top25_tables <- create_top25_tables(
  results = results,
  output_path = output_path,
  folder = "cross-sectional",
  method = "cross-sectional",
  p = "biomarker pvalue",
  beta = "biomarker coeff"
)

top25_df <- top25_tables$top25_df
top25_df_low <- top25_tables$top25_df_low


# /////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////
## Top 25 proteins forest plot  ----------------
# Cross-sectional baseline biomarker associations
# /////////////////////////////////////////////////////////
pval_thresold <- 0.05

# ..............................................................................
# Proteins to exclude
# ..............................................................................
exclude_proteins <- c(
  "pTau181",
  "pTau231",
  "pTau217",
  "AB38",
  "NEFL",
  "GFAP",
  "AB40",
  "lumipulse_ptau217",
  "APOE4"
)

# Remove excluded proteins
results_filtered <- results %>%
  filter(!biomarker %in% exclude_proteins)

# .............................................................................
# Prepare results with p < 0.1
# .............................................................................
plot_df <- results_filtered %>%
  filter(!is.na(`biomarker coeff`),
         !is.na(`biomarker std`),
         !is.na(`biomarker pvalue`)) %>%
  mutate(
    ci_low = `biomarker coeff` - 1.96 * `biomarker std`,
    ci_high = `biomarker coeff` + 1.96 * `biomarker std`,
    sig = case_when(
      `biomarker pvalue` < 0.05 ~ "p < 0.05",
      `biomarker pvalue` < 0.1  ~ "0.05 ≤ p < 0.1",
      TRUE ~ "NS"
    )
  ) %>%
  filter(`biomarker pvalue` < pval_thresold) %>%
  arrange(`biomarker pvalue`, desc(abs(`biomarker coeff`))) %>%
  arrange(`biomarker coeff`) %>%
  mutate(
    biomarker = factor(biomarker, levels = biomarker)
  )

# .............................................................................
# Forest-style plot
# .............................................................................
p_forest <- ggplot(plot_df, aes(x = `biomarker coeff`, y = biomarker, color = sig)) +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.5, color = "grey40") +
  geom_errorbarh(
    aes(xmin = ci_low, xmax = ci_high),
    height = 0.18,
    linewidth = 0.6,
    color = "grey45"
  ) +
  geom_point(size = 2.8) +
  scale_color_manual(
    values = c(
      "p < 0.05" = "#1f78b4",
      #"0.05 ≤ p < 0.1" = "#6baed6",
      "NS" = "grey70"
    ),
    drop = FALSE
  ) +
  labs(
    title = "Top baseline biomarkers associated with baseline cognition",
    x = expression(beta ~ "coefficient (per 1 SD higher biomarker, NPQ)"),
    y = NULL,
    color = NULL
  ) +
  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(size = 11),
    axis.text.y = element_text(size = 11),
    axis.text.x = element_text(size = 10),
    legend.position = "top"
  )

print(p_forest)

# .............................................................................
# Save figure and plotted table
# .............................................................................
ggsave(
  filename = paste0(output_path, "cross-sectional/forestplot_top25_cross_sectional_baseline_biomarkers_bmi_adj_mitt",pval_thresold,".png"),
  plot = p_forest,
  width = 8,
  height = 7,
  dpi = 600
)

ggsave(
  filename = paste0(output_path, "cross-sectional/forestplot_top25_cross_sectional_baseline_biomarkers_bmi_adj_mitt",pval_thresold,".svg"),
  plot = p_forest,
  device = svglite::svglite,
  width = 8,
  height = 7
)

write_csv(
  plot_df,
  paste0(output_path, "cross-sectional/forestplot_top25_cross_sectional_baseline_biomarkers_table_bmi_adj_pval_mitt",pval_thresold,".csv")
)

# /////////////////////////////////////////////////////////
## Volcano plot  ----------------
# Cross-sectional baseline biomarker associations
# /////////////////////////////////////////////////////////

plot_volcano(
  output_path = output_path,
  folder = "cross-sectional",
  output_file_name = "volcano_cross_sectional_baseline_biomarkers_imputed_bmi_adj_mitt",
  title = "Volcano plot of baseline biomarker associations with baseline cognition",
  results_filtered = results_filtered,
  top25_df = top25_df,
  p = "biomarker pvalue",
  beta = "biomarker coeff" 
)




# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
# Effect of Baseline BM on ztotalcomp. over time ----------------
# Adjusted with group and group x time ------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////

# Purpose:
# associations between baseline biomarkers and cognition
# over time in the total cohort, adjuted by group.
#
#
# Outcome:
#   ztotal_composite measured at:
#     - baseline: ztotal_composite_BL2
#     - 1 year:   ztotal_composite_12kk
#     - 2 years:  ztotal_composite_24kk
#
# Predictor:
#   Baseline biomarker level
#
# Main term of interest:
#   bm:year
#
# Model:
#   ztotal_composite ~ year * baseline biomarker + year * group +
#     age + sex + education + BMI + site
#   (group main effect AND group x time: each arm has its own cognitive slope,
#    so bm:year is adjusted for the intervention effect on cognitive change)
#     (1 + year | ID)
#
# Specifications:
# - randomization group (reference = control, 3) included as main effect
#   and as group x time interaction
# - (filtering on non-missing group retained)
#   the same eligible cohort definition
# - random intercept + random slope for year: (1 + year | ID)
# - campid reference = Helsinki (101)
# - sex reference = man (1)
# - winsorization at 1% and 99%
# - z-scaling 
# - remove patients without:
#     baseline biomarker
#     baseline cognition
#     at least one follow-up cognition
#     age, sex, education, BMI, site, or intervention group
#
# Output:
# One row per biomarker with beta, SE, raw p-value, and FDR.


# .............................................................................
# File path
# .............................................................................
file_path <- paste0(processed_path,"dataset_original_clean.csv")

# .............................................................................
# Load data
# .............................................................................
df_all <- read_csv(file_path, show_col_types = FALSE)


# .............................................................................
# Identify biomarkers
# .............................................................................
nulisa_0_cols <- names(df_all)[grepl("^nulisa_.*_0$", names(df_all))]
nulisa_2_cols <- names(df_all)[grepl("^nulisa_.*_2$", names(df_all))]

nulisa_bases <- str_remove(nulisa_0_cols, "_0$")
nulisa_bases <- nulisa_bases[paste0(nulisa_bases, "_2") %in% nulisa_2_cols]

extra_bases <- character(0)
if (all(c("lumipulse_ptau217_0", "lumipulse_ptau217_2") %in% names(df_all))) {
  extra_bases <- "lumipulse_ptau217"
}

biomarker_names <- c(nulisa_bases, extra_bases)
biomarker_names <- drop_excluded(biomarker_names)

# .............................................................................
# Common variables
# .............................................................................
base_vars <- c(
  "ID",
  "sub_sample_id",
  "age_bl1",
  "sex",
  "educ_years_BL1",
  "bmi_bl2",
  "campid",
  "ztotal_composite_BL2",
  "ztotal_composite_12kk",
  "ztotal_composite_24kk"
)

# .............................................................................
# 6. One-biomarker model runner
# .............................................................................
run_one_biomarker <- function(bm_base, data) {
  
  bm0 <- paste0(bm_base, "_0")
  
  cat("Running:", bm_base, "\n")
  
  df_model <- data %>%
    select(all_of(c(base_vars, bm0))) %>%
    filter(
      !is.na(.data[[bm0]]),
      !is.na(ztotal_composite_BL2),
      (!is.na(ztotal_composite_12kk) | !is.na(ztotal_composite_24kk)),
      !is.na(age_bl1),
      !is.na(sex),
      !is.na(educ_years_BL1),
      !is.na(bmi_bl2),
      !is.na(campid),
      !is.na(sub_sample_id)
    ) %>%
    mutate(
      bm0_w = winsorize(.data[[bm0]], p = 0.01)
    )
  
  if ((needs_log(bm_base) && any(df_model$bm0_w <= 0, na.rm = TRUE))) {
    return(tibble(
      biomarker = str_remove(bm_base, "^nulisa_"),
      n_ids = n_distinct(df_model$ID),
      n_obs = NA_integer_,
      error = "Non-positive baseline values after winsorization; log not possible"
    ))
  }
  
  df_model <- df_model %>%
    mutate(
      bm0_log = log_transformation(bm0_w, bm_base),
      bm0_log_z = zscore_safe(bm0_log)
    ) %>%
    filter(!is.na(bm0_log_z))
  
  df_long <- df_model %>%
    pivot_longer(
      cols = c(
        ztotal_composite_BL2,
        ztotal_composite_12kk,
        ztotal_composite_24kk
      ),
      names_to = "timepoint",
      values_to = "ztotal_composite"
    ) %>%
    mutate(
      year = case_when(
        timepoint == "ztotal_composite_BL2"  ~ 0,
        timepoint == "ztotal_composite_12kk" ~ 1,
        timepoint == "ztotal_composite_24kk" ~ 2
      ),
      sex = factor(sex),
      campid = factor(campid),
      sub_sample_id = factor(sub_sample_id),
      bmi_bl2_z = as.numeric(scale(bmi_bl2))
    ) %>%
    filter(!is.na(ztotal_composite))
  
  if ("1" %in% levels(df_long$sex)) {
    df_long$sex <- relevel(df_long$sex, ref = "1")
  }
  
  if ("3" %in% levels(df_long$sub_sample_id)) {
    df_long$sub_sample_id <- relevel(df_long$sub_sample_id, ref = "3")
  }
  if ("101" %in% levels(df_long$campid)) {
    df_long$campid <- relevel(df_long$campid, ref = "101")
  }
  
  model <- tryCatch(
    lmer(
      ztotal_composite ~ year * bm0_log_z +
        year * sub_sample_id +
        age_bl1 + sex + educ_years_BL1 + bmi_bl2_z + campid +
        (1 + year | ID),
      data = df_long,
      REML = FALSE,
      control = lmerControl(
        optimizer = "bobyqa",
        optCtrl = list(maxfun = 100000)
      )
    ),
    error = function(e) e
  )
  
  if (inherits(model, "error")) {
    return(tibble(
      biomarker = str_remove(bm_base, "^nulisa_"),
      n_ids = n_distinct(df_model$ID),
      n_obs = NA_integer_,
      error = model$message
    ))
  }
  
  fixed <- tidy(model, effects = "fixed") %>%
    select(term, estimate, std.error, p.value)
  
  tibble(
    biomarker = str_remove(bm_base, "^nulisa_"),
    n_ids = n_distinct(df_long$ID),
    n_obs = nrow(df_long),
    
    `Intercept coeff` = get_term_value(fixed, "(Intercept)", estimate),
    `Intercept std` = get_term_value(fixed, "(Intercept)", std.error),
    `Intercept pvalue` = get_term_value(fixed, "(Intercept)", p.value),
    
    `bm coeff` = get_term_value(fixed, "bm0_log_z", estimate),
    `bm std` = get_term_value(fixed, "bm0_log_z", std.error),
    `bm pvalue` = get_term_value(fixed, "bm0_log_z", p.value),
    
    `year coeff` = get_term_value(fixed, "year", estimate),
    `year std` = get_term_value(fixed, "year", std.error),
    `year pvalue` = get_term_value(fixed, "year", p.value),
    
    `bm:year coeff` = get_term_value(fixed, "year:bm0_log_z", estimate),
    `bm:year std` = get_term_value(fixed, "year:bm0_log_z", std.error),
    `bm:year pvalue` = get_term_value(fixed, "year:bm0_log_z", p.value),
    
    `age_0y coeff` = get_term_value(fixed, "age_bl1", estimate),
    `age_0y std` = get_term_value(fixed, "age_bl1", std.error),
    `age_0y pvalue` = get_term_value(fixed, "age_bl1", p.value),
    
    `bmi_0y coeff` = get_term_value(fixed, "bmi_bl2_z", estimate),
    `bmi_0y std` = get_term_value(fixed, "bmi_bl2_z", std.error),
    `bmi_0y pvalue` = get_term_value(fixed, "bmi_bl2_z", p.value),
    
    `sex coeff` = get_term_value(fixed, "sex2", estimate),
    `sex std` = get_term_value(fixed, "sex2", std.error),
    `sex pvalue` = get_term_value(fixed, "sex2", p.value),
    
    `educ_years_0y coeff` = get_term_value(fixed, "educ_years_BL1", estimate),
    `educ_years_0y std` = get_term_value(fixed, "educ_years_BL1", std.error),
    `educ_years_0y pvalue` = get_term_value(fixed, "educ_years_BL1", p.value),
    
    `sub_sample_id coeff` = get_term_value(fixed, "sub_sample_id2", estimate),
    `sub_sample_id std` = get_term_value(fixed, "sub_sample_id2", std.error),
    `sub_sample_id pvalue` = get_term_value(fixed, "sub_sample_id2", p.value),
    
    `group:year coeff` = get_term_value(fixed, "year:sub_sample_id2", estimate),
    `group:year std` = get_term_value(fixed, "year:sub_sample_id2", std.error),
    `group:year pvalue` = get_term_value(fixed, "year:sub_sample_id2", p.value),
    
    error = NA_character_
  )
}

# .............................................................................
# 7. Run all models
# .............................................................................
results_all <- map_dfr(biomarker_names, run_one_biomarker, data = df_all)

# .............................................................................
# 8. Add FDR columns
# .............................................................................
p_cols <- grep("pvalue$", names(results_all), value = TRUE)

for (pcol in p_cols) {
  fdr_col <- paste0(pcol, " FDR")
  results_all[[fdr_col]] <- p.adjust(results_all[[pcol]], method = "fdr")
}

# .............................................................................
# 9. Round numeric columns
# .............................................................................
results_all <- results_all %>%
  mutate(across(where(is.numeric), ~ round(.x, 6)))



# .............................................................................
# Save results
# .............................................................................
save_path <- paste0(output_path, "Longitudinal/ztotal_composite_change_mixed_model_baseline_bm_total_cohort_adj_group_x_time.csv")
write_csv(
  results_all,
  save_path
)




# .............................................................................
# Top bm 
# .............................................................................

top25_tables <- create_top25_tables(
  results = results_all,
  output_path = output_path,
  folder = "Longitudinal",
  method = "ztotal_vs_time_totalcohort_adj_group_x_time",
  p = "bm:year pvalue",
  beta = "bm:year coeff"
)

top25_df <- top25_tables$top25_df
top25_df_low <- top25_tables$top25_df_low


# /////////////////////////////////////////////////////////
## Volcano plot  ----------------
# /////////////////////////////////////////////////////////
exclude_proteins <- c(
  "pTau181",
  "pTau231",
  "pTau217",
  "AB38",
  "NEFL",
  "GFAP",
  "AB40",
  "lumipulse_ptau217",
  "APOE4"
)

# Remove excluded proteins
results_filtered <- results_all %>%
  filter(!biomarker %in% exclude_proteins)

plot_volcano(
  output_path = output_path,
  folder = "Longitudinal",
  output_file_name = "volcano_ztotal_vs_time_totalcohort_adj_group_x_time",
  title = "Volcano plot of baseline biomarker associations with cognition \n over time - adjusted for group and group x time",
  results_filtered = results_filtered,
  top25_df = top25_df,
  p = "bm:year pvalue",
  beta = "bm:year coeff"
  
)


# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
# Difference in biomarkers by APOE ----------------------------------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////




# .............................................................................
# File path
# .............................................................................
file_path <- paste0(processed_path,"dataset_original_clean.csv")


# .............................................................................
# Load data
# .............................................................................
df_all <- read_csv(file_path, show_col_types = FALSE)


# .............................................................................
# Columns
# .............................................................................
bm_list_prot = c('nulisa_HTT_0', 'nulisa_TARDBP_0', 'nulisa_SNCA_0','nulisa_SOD1_0')
base_vars <- c(
  "ID",
  "apoe4_status"
  
)
df <- df_all %>%
  filter(
    (!is.na(ztotal_composite_12kk)| !is.na(ztotal_composite_24kk))) %>%
  select(all_of(c(base_vars, bm_list_prot))) 




# Define the variables
bm_list_prot <- c('nulisa_HTT_0', 'nulisa_TARDBP_0', 'nulisa_SNCA_0', 'nulisa_SOD1_0')
base_vars <- c("ID", "apoe4_status")

# Select the relevant columns
df <- df_all %>%
  select(all_of(c(base_vars, bm_list_prot)))

# Initialize a results dataframe
results_all <- tibble(
  Protein = character(),
  Test = character(),
  Statistic = numeric(),
  P.Value = numeric(),
  Conclusion = character(),
  Mean_Carrier = character(),
  Mean_NonCarrier = character()
)

# Loop through each protein
for (protein in bm_list_prot) {
  # Perform ANOVA
  anova_result <- perform_anova(df, protein, "apoe4_status")
  
  # Extract the row for the group variable
  anova_row <- anova_result %>%
    filter(term == "apoe4_status")
  
  # Calculate mean and std for each group
  mean_std_df <- calculate_mean_std(df, protein, "apoe4_status")
  
  # Extract mean (std) for carriers and non-carriers
  mean_carrier <- mean_std_df %>%
    filter(apoe4_status == "Carrier") %>%
    pull(mean_std) %>%
    as.character()
  
  mean_non_carrier <- mean_std_df %>%
    filter(apoe4_status == "Non-Carrier") %>%
    pull(mean_std) %>%
    as.character()
  
  # Determine conclusion
  conclusion <- ifelse(
    anova_row$p.value < 0.05,
    "Significant",
    "Not Significant"
  )
  
  # Append results
  results_all <- bind_rows(
    results_all,
    tibble(
      Protein = protein,
      Test = "ANOVA",
      Statistic = anova_row$statistic,
      P.Value = anova_row$p.value,
      Conclusion = conclusion,
      Mean_Carrier = ifelse(is.null(mean_carrier), NA, mean_carrier),
      Mean_NonCarrier = ifelse(is.null(mean_non_carrier), NA, mean_non_carrier)
    )
  )
}





save_path <- paste0(output_path, "cross-sectional/apoe_diff.csv")
write_csv(
  results_all,
  save_path
)

