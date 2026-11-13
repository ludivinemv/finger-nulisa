# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
# Script for Nulisa biomarkers analyses: intervention vs control
# NULISA NPQ used as is (already log2); Lumipulse pTau217 log-transformed 
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////


Sys.setlocale("LC_ALL", "en_US.UTF-8")





# /////////////////////////////////////////////////////////////////////////////
# Arguments -------------------------------------------------------------------
# /////////////////////////////////////////////////////////////////////////////

## Options ----
#user name
user <- "makrina" # "makrina" or "ludivine"
version <- "2-1"

group_name <- "intervention" # adherence analyses removed from this version


color_carrier ="#0072B2"
color_noncarrier= "#E69F00"
color_control="#4D4D4D"
color_interv = "#CC79A7"
color_low = "#B0DDEB"
color_male = "#009E73"
color_female = "#D55E00"






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
              "tidyverse",
              "colorblindr",
              "effectsize",
              "pheatmap",
              "parameters",
              "ragg",
              "svglite")


# Loop to install packages
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) { # Install only if the package is not already installed
    if (pkg == "colorblindr"){
      remotes::install_github("clauswilke/colorblindr")
    }else{
      install.packages(pkg)
    }
  }
}


# Loop to load packages
for (pkg in packages) {
  library(pkg, character.only = TRUE)
}

# remove unused packages used to load other packages
gc()


# ..............................................................................
## Variables definition  -------------------------------------------------------
# ..............................................................................
var_group <-  "sub_sample_id"
ref_group <- "sub_sample_id2"
interv_val_group <- "intervention"
test_group_val <-"Intervention"
ref_control = "3"


list_colors <- c(
  "male" = color_male,
  "female" = color_female,
  "carrier" = color_carrier,
  "noncarrier" = color_noncarrier,
  "interv" = color_interv,
  "control" = color_control,
  "low" = color_low
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
original_path <- paste0(folder_path,"original/")
processed_path <- paste0(folder_path,"processed/")
output_path <- paste0(folder_path,"outputs_R/",group_name,"/")


# Check that the input data file exists before running anything
data_file <- paste0(processed_path, "dataset_original.csv")
if (!file.exists(data_file)) {
  stop("Input file not found:\n", data_file,
       "\nCheck folder_path / processed_path (or run the total-cohort script first, which creates it).")
}

if (!dir.exists(output_path)) {dir.create(output_path, recursive = TRUE)}# Create the folder if it does not exist
for (fold in c("Longitudinal","cross-sectional","Stratified","Stratified_sex")){
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
# Effect of Baseline BM and group on ztotalcomp. over time ----------------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
# produce Longitudinal/ztotal_composite_change_mixed_model_baseline_bm_log_z_imputed_zeros_bmi_adj.csv


# ========================================================= \n
# ========================================================= \n
# 2-way BBM x group class x time \n
# ========================================================= \n
# ========================================================= \n
# 
# Mixed-effects models for all baseline NULISA biomarkers \n
#  + lumipulse_ptau217 \n
#  Output: one row per biomarker, columns = beta / SE / raw p / FDR\n
# \n
#  Specifications:\n
#  - random intercept + random slope for year: (1 + year | ID)\n
#  - campid reference = Helsinki (101)\n
#  - group reference = control \n
#  - sex reference = man (1)\n
#  - winsorization at 1% and 99%, baseline and 2-year separately\n
#  - log transformation AFTER winsorization\n
#  - z-scaling AFTER log transformation\n
#  - remove patients without baseline biomarker\n
#  - remove patients without baseline cognition OR without at least one follow-up\n
#  - remove patients without one of the cofactors\n")


# .............................................................................
# File path
# .............................................................................
file_path <- paste0(processed_path,"dataset_original.csv")
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
  var_group,
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
# One-biomarker model runner
# .............................................................................
run_one_biomarker <- function(bm_base, data, var_group) {
  
  bm0 <- paste0(bm_base, "_0")
  bm2 <- paste0(bm_base, "_2")
  
  
  #cat("Running:", bm_base, "\n") 
  
  df_model <- data %>%
    select(all_of(c(base_vars, bm0, bm2))) %>%
    filter(
      !is.na(.data[[bm0]]),
      !is.na(ztotal_composite_BL2),
      (!is.na(ztotal_composite_12kk) | !is.na(ztotal_composite_24kk)),
      !is.na(age_bl1),
      !is.na(sex),
      !is.na(educ_years_BL1),
      !is.na(bmi_bl2),
      !is.na(campid),
      !is.na(.data[[var_group]])
    ) %>%
    mutate(
      bm0_w = winsorize(.data[[bm0]], p = 0.01),
      bm2_w = winsorize(.data[[bm2]], p = 0.01)
    )

  # Log requires positive values at both timepoints
  if ((needs_log(bm_base) && any(df_model$bm0_w <= 0, na.rm = TRUE)) ||
      (needs_log(bm_base) && any(df_model$bm2_w <= 0, na.rm = TRUE))) {
    return(tibble(
      biomarker = str_remove(bm_base, "^nulisa_"),
      n_ids = n_distinct(df_model$ID),
      n_obs = NA_integer_,
      error = "Non-positive values after winsorization; log not possible"
    ))
  }
  
  df_model <- df_model %>%
    mutate(
      bm0_log   = transform_bm(bm0_w, bm_base),
      bm2_log   = transform_bm(bm2_w, bm_base),
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
      !!var_group  := factor(.data[[var_group]]),
      sex = factor(sex),
      campid = factor(campid),
      bmi_bl2_z = as.numeric(scale(bmi_bl2))
    ) %>%
    filter(!is.na(ztotal_composite))
  
  
  # Reference groups
  if ("3" %in% levels(df_long$sub_sample_id)) {
    df_long$sub_sample_id <- relevel(df_long$sub_sample_id, ref = "3")
  }

  if ("1" %in% levels(df_long$sex)) {
    df_long$sex <- relevel(df_long$sex, ref = "1")
  }
  if ("101" %in% levels(df_long$campid)) {
    df_long$campid <- relevel(df_long$campid, ref = "101")
  }
  

  
  model <- tryCatch(
    lmer(
      paste0(
        "ztotal_composite ~ year * bm0_log_z * ", var_group,
        " + age_bl1 + sex + educ_years_BL1 + bmi_bl2_z + campid +",
        " (1 + year | ID)"
      ),
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
    dplyr::select(term, estimate, std.error, p.value)
    # .............................................................................
  # Accurate group-specific biomarker × time slopes
  # .............................................................................
  
  bm_group_int     <- paste0("bm0_log_z:", ref_group)
  year_group_int   <- paste0("year:", ref_group)
  bm_year_int      <- "year:bm0_log_z"
  bm_year_group_int <- paste0("year:bm0_log_z:", ref_group)  
  
  control_slope <- get_linear_combo(
    model,
    terms = c("year:bm0_log_z")
  )
  
  intervention_slope <- get_linear_combo(
    model,
    terms = c(
      bm_year_int,
      bm_year_group_int
    )
  )
  tibble(
  biomarker = str_remove(bm_base, "^nulisa_"),
  n_ids = n_distinct(df_long$ID),
  n_obs = nrow(df_long),

  `Intercept coeff` = get_term_value(fixed, "(Intercept)", estimate),
  `Intercept std`   = get_term_value(fixed, "(Intercept)", std.error),
  `Intercept pvalue`= get_term_value(fixed, "(Intercept)", p.value),

  `group coeff` = get_term_value(fixed, ref_group, estimate),
  `group std`   = get_term_value(fixed, ref_group, std.error),
  `group pvalue`= get_term_value(fixed, ref_group, p.value),

  `bm coeff` = get_term_value(fixed, "bm0_log_z", estimate),
  `bm std`   = get_term_value(fixed, "bm0_log_z", std.error),
  `bm pvalue`= get_term_value(fixed, "bm0_log_z", p.value),

  `group:bm coeff` = get_term_value(fixed, bm_group_int, estimate),
  `group:bm std`   = get_term_value(fixed, bm_group_int, std.error),
  `group:bm pvalue`= get_term_value(fixed, bm_group_int, p.value),

  `year coeff` = get_term_value(fixed, "year", estimate),
  `year std`   = get_term_value(fixed, "year", std.error),
  `year pvalue`= get_term_value(fixed, "year", p.value),

  `group:year coeff` = get_term_value(fixed, year_group_int, estimate),
  `group:year std`   = get_term_value(fixed, year_group_int, std.error),
  `group:year pvalue`= get_term_value(fixed, year_group_int, p.value),

  `bm:year coeff` = get_term_value(fixed, bm_year_int, estimate),
  `bm:year std`   = get_term_value(fixed, bm_year_int, std.error),
  `bm:year pvalue`= get_term_value(fixed, bm_year_int, p.value),

  `group:bm:year coeff` = get_term_value(fixed, bm_year_group_int, estimate),
  `group:bm:year std`   = get_term_value(fixed, bm_year_group_int, std.error),
  `group:bm:year pvalue`= get_term_value(fixed, bm_year_group_int, p.value),

    `control bm:year coeff` = control_slope$estimate,
    `control bm:year std` = control_slope$std.error,
    `control bm:year pvalue` = control_slope$p.value,
    `control bm:year ci_low` = control_slope$ci_low,
    `control bm:year ci_high` = control_slope$ci_high,
    
    !!glue::glue("{interv_val_group} bm:year coeff") := intervention_slope$estimate,
    !!glue::glue("{interv_val_group} bm:year std") := intervention_slope$std.error,
    !!glue::glue("{interv_val_group} bm:year pvalue") := intervention_slope$p.value,
    !!glue::glue("{interv_val_group} bm:year ci_low") := intervention_slope$ci_low,
    !!glue::glue("{interv_val_group} bm:year ci_high") := intervention_slope$ci_high,

    `age_0y coeff` = get_term_value(fixed, "age_bl1", estimate),
    `age_0y std`   = get_term_value(fixed, "age_bl1", std.error),
    `age_0y pvalue`= get_term_value(fixed, "age_bl1", p.value),
    
    `bmi_0y coeff` = get_term_value(fixed, "bmi_bl2_z", estimate),
    `bmi_0y std`   = get_term_value(fixed, "bmi_bl2_z", std.error),
    `bmi_0y pvalue`= get_term_value(fixed, "bmi_bl2_z", p.value),
    
    
    `sex coeff` = get_term_value(fixed, "sex2", estimate),
    `sex std`   = get_term_value(fixed, "sex2", std.error),
    `sex pvalue`= get_term_value(fixed, "sex2", p.value),
    
    `educ_years_0y coeff` =
      get_term_value(fixed, "educ_years_BL1", estimate),
    `educ_years_0y std` =
      get_term_value(fixed, "educ_years_BL1", std.error),
    `educ_years_0y pvalue` =
      get_term_value(fixed, "educ_years_BL1", p.value),
    
    error = NA_character_
  )
  }
# .............................................................................
# Run all models
# .............................................................................
results_all <- map_dfr(biomarker_names, run_one_biomarker, data = df_all, var_group = var_group, .progress = interactive())




# .............................................................................
# Add FDR columns
# .............................................................................
p_cols <- grep("pvalue$", names(results_all), value = TRUE)

for (pcol in p_cols) {
  fdr_col <- paste0(pcol, " FDR")
  results_all[[fdr_col]] <- p.adjust(results_all[[pcol]], method = "fdr")
}

# .............................................................................
# Round numeric columns
# .............................................................................
results_all <- results_all %>%
  mutate(across(where(is.numeric), ~ round(.x, 6)))

# .............................................................................
# Save results
# .............................................................................
save_path <- paste0(output_path, "Longitudinal/ztotal_composite_change_mixed_model_baseline_bm_log_z_imputed_zeros_bmi_adj.csv")
write_csv(
  results_all,
  save_path
)

cat("Saved under :")
cat(save_path)



# .............................................................................
## Top bm ----
# .............................................................................

top25_tables <- create_top25_tables(
  results = results_all,
  output_path = output_path,
  folder = "Longitudinal",
  method = "bm_year",
  p = "group:bm:year pvalue"
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
  output_file_name = "volcano_ztotal_vs_time_vs_bbm_vs_group",
  title = paste0("Volcano plot of baseline biomarker associations with cognition \n over time and ",test_group_val," group"),
  results_filtered = results_filtered,
  top25_df = top25_df,
  p = "group:bm:year pvalue",
  beta = "group:bm:year coeff"
  
)


# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
## Create PDF with 6 bm per page -----------------------------------------------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////

# produce "Longitudinal/all_biomarker_diagnostics_6_per_page_imputed_zeros_bmi_adj.pdf"
#   Compact diagnostics PDF
#   6 biomarkers per page
#   Each biomarker:
#     1) Residuals vs Fitted
#     2) QQ plot



# .............................................................................
# File path
# .............................................................................
file_path <- paste0(processed_path,"dataset_original.csv")
# .............................................................................
# Load data
# .............................................................................
df_all <- read_csv(file_path, show_col_types = FALSE)
# .............................................................................
# Biomarker names
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
  var_group,
  "age_bl1",
  "sex",
  "educ_years_BL1",
  "campid",
  "bmi_bl2",
  "ztotal_composite_BL2",
  "ztotal_composite_12kk",
  "ztotal_composite_24kk"
)

# .............................................................................
# Fit one model
#    Uses:
#    baseline cognition + at least one FU
#    winsorize -> log -> z-score
# .............................................................................
run_one_model <- function(bm_base, data, var_group) {
  

  bm0 <- paste0(bm_base, "_0")
  bm2 <- paste0(bm_base, "_2")

  
  df_model <- data %>%
    select(all_of(c(base_vars, bm0, bm2))) %>%
    filter(
      !is.na(.data[[bm0]]),
      !is.na(.data[[bm2]]),
      !is.na(ztotal_composite_BL2),
      (!is.na(ztotal_composite_12kk) | !is.na(ztotal_composite_24kk)),
      !is.na(age_bl1),
      !is.na(sex),
      !is.na(educ_years_BL1),
      !is.na(campid),
      !is.na(bmi_bl2),
      !is.na(.data[[var_group]])
    ) %>%
    mutate(
      bm0_w = winsorize(.data[[bm0]], p = 0.01),
      bm2_w = winsorize(.data[[bm2]], p = 0.01)
    )
  
  if ((needs_log(bm_base) && any(df_model$bm0_w <= 0, na.rm = TRUE)) ||
      (needs_log(bm_base) && any(df_model$bm2_w <= 0, na.rm = TRUE))) return(NULL)
  
  df_model <- df_model %>%
    mutate(
      bm0_log   = transform_bm(bm0_w, bm_base),
      bm0_log_z = zscore_safe(bm0_log)
    ) %>%
    filter(!is.na(bm0_log_z))
  
  if (n_distinct(df_model$ID) < 20) return(NULL)
  
  df_long <- df_model %>%
    pivot_longer(
      cols = c(
        ztotal_composite_BL2,
        ztotal_composite_12kk,
        ztotal_composite_24kk
      ),
      names_to = "timepoint",
      values_to = "ztotal_composite"
    ) 
  df_long <- df_long %>%
    mutate(
      year = case_when(
        timepoint == "ztotal_composite_BL2"  ~ 0,
        timepoint == "ztotal_composite_12kk" ~ 1,
        timepoint == "ztotal_composite_24kk" ~ 2
      ),
      !!rlang::sym(var_group)  := factor(df_long[[var_group]]),
      sex = factor(sex),
      campid = factor(campid),
      bmi_bl2_z = as.numeric(scale(bmi_bl2))
    ) %>%
    filter(!is.na(ztotal_composite))
  
  if ("3" %in% levels(df_long$sub_sample_id)) {
    df_long$sub_sample_id <- relevel(df_long$sub_sample_id, ref = "3")
  }
  if ("1" %in% levels(df_long$sex)) {
    df_long$sex <- relevel(df_long$sex, ref = "1")
  }
  if ("101" %in% levels(df_long$campid)) {
    df_long$campid <- relevel(df_long$campid, ref = "101")
  }
  
  model <- tryCatch(
    lmer(
      
      paste0(
        "ztotal_composite ~ year * bm0_log_z * ", var_group,
        " + age_bl1 + sex + educ_years_BL1 + bmi_bl2_z + campid +",
        " (1 + year | ID)"
      ),
      data = df_long,
      REML = FALSE,
      control = lmerControl(
        optimizer = "bobyqa",
        optCtrl = list(maxfun = 100000)
      )
    ),
    error = function(e) NULL
  )
  if (is.null(model)) return(NULL)
  list(
    biomarker = str_remove(bm_base, "^nulisa_"),
    model = model
  )
}
# .............................................................................
# Fit all models
# .............................................................................
models <- map(biomarker_names, run_one_model, data = df_all, var_group = var_group, .progress = interactive())
models <- models[!map_lgl(models, is.null)]

cat("Models fitted:", length(models), "\n")

# .............................................................................
# Save compact diagnostics PDF
#    6 biomarkers per page
# .............................................................................
pdf(
  paste0(output_path,"Longitudinal/all_biomarker_diagnostics_6_per_page_imputed_zeros_bmi_adj.pdf"),
  width = 12,
  height = 14
)

plots_per_page <- 6
n_models <- length(models)

for (i in seq(1, n_models, by = plots_per_page)) {
  
  idx <- i:min(i + plots_per_page - 1, n_models)
  
  par(
    mfrow = c(length(idx), 2),
    mar = c(3, 3, 2, 1),
    oma = c(0, 0, 2, 0)
  )
  
  for (j in idx) {
    
    model <- models[[j]]$model
    bm <- models[[j]]$biomarker
    
    res <- resid(model)
    fit <- fitted(model)
    
    # .................................
    # Residuals vs fitted
    # .................................
    plot(
      fit, res,
      pch = 16,
      cex = 0.5,
      main = paste(bm, "- Resid vs Fit"),
      xlab = "Fitted",
      ylab = "Residuals"
    )
    abline(h = 0, col = "red", lty = 2)
    
    # .................................
    # QQ plot
    # .................................
    qqnorm(
      res,
      pch = 16,
      cex = 0.5,
      main = paste(bm, "- QQ")
    )
    qqline(res, col = "red")
  }
}

dev.off()

cat("Saved: all_biomarker_diagnostics_6_per_page_imputed_zeros_bmi_adj.pdf\n")







# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
## Two-panel 3-way interaction figure + supplementary table --------------------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////

# Two-panel 3-way interaction figure + supplementary table
#
# Panel A:
#   Formal group x biomarker x time 
#
# Panel B:
#   Exact group-specific slopes:
#   Control = biomarker x time
#   intervention = control slope + group x biomarker x time
#
# Main figure:
#   biomarkers with raw 3-way p < 0.05
#
# Supplementary table:
#   biomarkers with raw 3-way p < 0.10

# produce 
# - paste0(output_path,"Longitudinal/Supplementary_Table_3way_group_biomarker_time_interactions_p010.csv")
# - paste0(output_path,"Longitudinal/mixed_models_3way_two_panel_interaction_and_slopes_p005.png"),
# - paste0(output_path,"Longitudinal/mixed_models_3way_two_panel_interaction_and_slopes_p005.pdf"),
# - paste0(output_path,"Longitudinal/mixed_models_3way_panelA_formal_interaction_p005.csv")
# - paste0(output_path,"Longitudinal/mixed_models_3way_panelB_group_specific_slopes_p005.csv")


# .............................................................................
# File path
# .............................................................................
file_path <- paste0(output_path, "Longitudinal/ztotal_composite_change_mixed_model_baseline_bm_log_z_imputed_zeros_bmi_adj.csv")

# .............................................................................
# Proteins to exclude
# .............................................................................
exclude_proteins <- c(
  "pTau181",
  "pTau231",
  "pTau217",
  "AB38",
  "NEFL",
  "GFAP",
  "AB40",
  "lumipulse_ptau217"
)

# .............................................................................
# Read data
# .............................................................................
res <- read_csv(file_path, show_col_types = FALSE)

if (!"Protein" %in% names(res)) {
  names(res)[1] <- "Protein"
}

res <- res %>%
  mutate(
    Protein = str_remove(Protein, "^nulisa_")
  ) %>%
  filter(!Protein %in% exclude_proteins)

# .............................................................................
# TWO-PANEL 3-WAY INTERACTION FIGURE
# .............................................................................


required_3way_cols <- c(
  "bm:year coeff",
  "bm:year std",
  "bm:year pvalue",
  
  "group:bm:year coeff",
  "group:bm:year std",
  "group:bm:year pvalue",
  
  "control bm:year coeff",
  "control bm:year pvalue",
  "control bm:year ci_low",
  "control bm:year ci_high",
  
  glue::glue("{interv_val_group} bm:year coeff"),
  glue::glue("{interv_val_group} bm:year pvalue"),
  glue::glue("{interv_val_group} bm:year ci_low"),
  glue::glue("{interv_val_group} bm:year ci_high")
)

missing_3way_cols <- setdiff(required_3way_cols, names(res))

if (length(missing_3way_cols) > 0) {
  
  warning(
    paste0(
      "3-way plot skipped because these columns are missing:\n",
      paste(missing_3way_cols, collapse = "\n"),
      "\n\nRe-run the mixed model script after adding exact control/Intervention group linear combinations."
    )
  )
  
} else {
  txt_strong_group = paste0("Stronger biomarker association in ",test_group_val, " group")
  # .............................................................................
  # Optional FDR column
  # .............................................................................
  threeway_fdr_col <- "group:bm:year pvalue FDR"
  has_threeway_fdr <- threeway_fdr_col %in% names(res)
  
  if (!has_threeway_fdr) {
    message("No 3-way FDR column found. Only raw p-value significance will be marked.")
  }
  
  # .............................................................................
  # Build wide table
  # .............................................................................
  plot_3way_wide_all <- res %>%
    dplyr::transmute(
      Protein,
      
      twoway_beta = `bm:year coeff`,
      twoway_se   = `bm:year std`,
      twoway_p    = `bm:year pvalue`,
      
      threeway_beta = `group:bm:year coeff`,
      threeway_se   = `group:bm:year std`,
      threeway_p    = `group:bm:year pvalue`,
      threeway_fdr  = if (has_threeway_fdr) res[[threeway_fdr_col]] else NA_real_,
      
      control_beta    = `control bm:year coeff`,
      control_p       = `control bm:year pvalue`,
      control_ci_low  = `control bm:year ci_low`,
      control_ci_high = `control bm:year ci_high`,
      
      intervention_beta    = res[[paste0(interv_val_group, " bm:year coeff")]],
      intervention_p       = res[[paste0(interv_val_group, " bm:year pvalue")]],
      intervention_ci_low  = res[[paste0(interv_val_group, " bm:year ci_low")]],
      intervention_ci_high = res[[paste0(interv_val_group, " bm:year ci_high")]]
    ) %>%
    dplyr::filter(
      !is.na(threeway_beta),
      !is.na(threeway_se),
      !is.na(threeway_p)
    ) %>%
    dplyr::mutate(
      threeway_ci_low  = threeway_beta - 1.96 * threeway_se,
      threeway_ci_high = threeway_beta + 1.96 * threeway_se,
      abs_threeway_beta = abs(threeway_beta),
      
      direction_block = dplyr::case_when(
        abs(twoway_beta + threeway_beta) >= abs(twoway_beta ) ~ txt_strong_group,
        abs(twoway_beta + threeway_beta) < abs(twoway_beta )  ~ "Stronger biomarker association in control"
      ),
      threeway_marker = dplyr::case_when(
        !is.na(threeway_fdr) & threeway_fdr < 0.05 ~ "**",
        threeway_p < 0.05 ~ "*",
        TRUE ~ ""
      ),
      
      interpretation = dplyr::case_when(
        threeway_beta > 0 ~
          paste0("Biomarker-cognition slope is more positive in ",test_group_val," than control"),
        threeway_beta < 0 ~
          paste0("Biomarker-cognition slope is more positive in control than ",test_group_val),
        TRUE ~
          "No directional difference"
      )
    )
  
  # .............................................................................
  # Supplementary table: p < 0.10
  # .............................................................................
  supp_3way_table <- plot_3way_wide_all %>%
    dplyr::filter(threeway_p < 0.10) %>%
    dplyr::arrange(threeway_p) %>%
    dplyr::mutate(
      significance_label = dplyr::case_when(
        !is.na(threeway_fdr) & threeway_fdr < 0.05 ~ "FDR significant",
        threeway_p < 0.05 ~ "Nominal p < 0.05",
        threeway_p < 0.10 ~ "Nominal p 0.05-0.10",
        TRUE ~ "Not significant"
      )
    ) %>%
    dplyr::select(
      Protein,
      threeway_beta,
      threeway_se,
      threeway_ci_low,
      threeway_ci_high,
      threeway_p,
      threeway_fdr,
      abs_threeway_beta,
      direction_block,
      interpretation,
      significance_label,
      control_beta,
      control_p,
      control_ci_low,
      control_ci_high,
      intervention_beta,
      intervention_p,
      intervention_ci_low,
      intervention_ci_high
    ) %>%
    dplyr::mutate(
      dplyr::across(where(is.numeric), ~ round(.x, 4))
    )
  
  readr::write_csv(
    supp_3way_table,
    paste0(output_path,"Longitudinal/Supplementary_Table_3way_group_biomarker_time_interactions_p010.csv")
  )
  
  # .............................................................................
  # Main figure table: p < 0.05
  # .............................................................................
  plot_3way_wide <- plot_3way_wide_all %>%
    dplyr::filter(threeway_p < 0.05) %>%
    dplyr::arrange(dplyr::desc(abs(threeway_beta)))
  
  if (nrow(plot_3way_wide) == 0) {
    stop("No biomarkers with raw 3-way p < 0.05 after filtering.")
  }
  
  protein_order_3way <- plot_3way_wide %>%
    dplyr::arrange(threeway_beta) %>%
    dplyr::pull(Protein)
  
  plot_3way_wide <- plot_3way_wide %>%
    dplyr::mutate(
      Protein = factor(Protein, levels = protein_order_3way)
    )
  
  
  # .............................................................................
  # Panel A: formal 3-way interaction
  # .............................................................................
  text_strong_interv = paste0("Stronger biomarker association in ",test_group_val, " group") 
  
  
  color_val = c(
    "Stronger biomarker association in control" = color_control,
    "Stronger biomarker association in Intervention group"= color_interv
  )
  
  p_A_3way <- ggplot(
    plot_3way_wide,
    aes(x = threeway_beta, y = Protein)
  ) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.55,
      color = "grey45"
    ) +
    geom_errorbarh(
      aes(xmin = threeway_ci_low, xmax = threeway_ci_high, color = direction_block),
      height = 0.22,
      linewidth = 1.05
    ) +
    geom_point(
      aes(color = direction_block),
      size = 4.0,
      shape = 16
    ) +
    geom_text(
      aes(
        x = dplyr::if_else(
          threeway_beta >= 0,
          threeway_ci_high + 0.004,
          threeway_ci_low - 0.004
        ),
        label = threeway_marker
      ),
      size = 5.2,
      fontface = "bold",
      show.legend = FALSE
    ) +
    scale_color_manual(
      values = color_val
    ) +
    coord_cartesian(
      xlim = c(-0.14, 0.14),
      clip = "off"
    ) +
    labs(
      title = paste0("A. ",test_group_val,"-dependent biomarker modification"),
      subtitle = "3-way coefficient: group x biomarker x time",
      x = "3-way beta coefficient",
      y = NULL,
      color = NULL
    ) +
    guides(
      color = guide_legend(nrow = 2, byrow = TRUE)
    ) +
    theme_classic(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 10.5, color = "grey35"),
      axis.text.y = element_text(size = 11),
      axis.text.x = element_text(size = 11),
      axis.title.x = element_text(size = 11),
      legend.position = "top",
      legend.justification = "left",
      legend.direction = "vertical",
      legend.margin = margin(0, 0, -3, 0),
      legend.box.spacing = unit(0.2, "lines"),
      legend.title.align = 0,
      legend.text.align = 0,
      legend.text = element_text(size = 9.5),
      plot.margin = margin(10, 25, 10, 10)
    )
  
  # .............................................................................
  # Panel B: control/Intervention group-specific slopes
  # .............................................................................
  plot_3way_long <- plot_3way_wide %>%
    tidyr::pivot_longer(
      cols = c(
        control_beta,
        control_p,
        control_ci_low,
        control_ci_high,
      intervention_beta,
      intervention_p,
      intervention_ci_low,
      intervention_ci_high
      ),
      names_to = c("Group", ".value"),
    names_pattern = "(control|intervention)_(beta|p|ci_low|ci_high)"
    ) %>%
    dplyr::mutate(
      Group = dplyr::recode(
        Group,
        control = "Control",
        intervention  = test_group_val
      ),
      Group = factor(Group, levels = c("Control", test_group_val)),
      subgroup_marker = dplyr::if_else(p < 0.05, "*", ""),
      label_x = dplyr::case_when(
        beta >= 0 ~ ci_high + 0.0015,
        beta < 0  ~ ci_low  - 0.0015
      )
    )
  


  color_val = c(
    "Control" = color_control,
    "Intervention" = color_interv
  )
  
  p_B_3way <- ggplot(
    plot_3way_long,
    aes(
      x = beta,
      y = Protein,
      color = Group
    )
  ) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.55,
      color = "grey45"
    ) +
    geom_errorbarh(
      aes(xmin = ci_low, xmax = ci_high),
      height = 0.24,
      linewidth = 0.90,
      position = position_dodge(width = 0.75)
    ) +
    geom_point(
      size = 3.0,
      shape = 16,
      position = position_dodge(width = 0.75)
    ) +
    geom_text(
      aes(x = label_x, label = subgroup_marker),
      position = position_dodge(width = 0.75),
      size = 4.3,
      fontface = "bold",
      show.legend = FALSE
    ) +
    scale_color_manual(
      values = color_val)+
    coord_cartesian(
      xlim = c(-0.08, 0.08),
      clip = "off"
    ) +
    labs(
      title = paste0("B. Control- and ",test_group_val,"-specific biomarker slopes"),
      subtitle = "Exact group-specific slopes estimated from mixed-effects models",
      x = "Estimated beta coefficient for cognitive change",
      y = NULL,
      color = "Group"
    ) +
    theme_classic(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 10.5, color = "grey35"),
      axis.text.y = element_text(color = "grey55", size = 9),
      axis.ticks.y = element_blank(),
      axis.text.x = element_text(size = 11),
      axis.title.x = element_text(size = 11),
      legend.position = "top",
      legend.justification = "left",
      legend.margin = margin(0, 0, -3, 0),
      legend.box.spacing = unit(0.2, "lines"),
      legend.title.align = 0,
      legend.text.align = 0,
      legend.text = element_text(size = 9),
      legend.title = element_text(size = 9),
      plot.margin = margin(10, 25, 10, 5)
    )
  
  # .............................................................................
  # Combine panels
  # .............................................................................
  combined_3way <- p_A_3way + p_B_3way +
    patchwork::plot_layout(
      ncol = 2,
      widths = c(0.95, 1.35)
    ) &
    theme(
      panel.spacing = unit(1.5, "lines")
    )
  
  print(combined_3way)
  
  # .............................................................................
  # Save
  # .............................................................................
  fig_height_3way <- max(9, 0.55 * nrow(plot_3way_wide))
  
  ggsave(
    filename = paste0(output_path,"Longitudinal/mixed_models_3way_two_panel_interaction_and_slopes_p005.png"),
    plot = combined_3way,
    width = 15,
    height = fig_height_3way,
    dpi = 600,
    bg = "white"
  )
  
  ggsave(
    filename = paste0(output_path,"Longitudinal/mixed_models_3way_two_panel_interaction_and_slopes_p005.svg"),
    plot = combined_3way,
    device = svglite::svglite,
    width = 15,
    height = fig_height_3way
  )
  
  ggsave(
    filename =  paste0(output_path,"Longitudinal/mixed_models_3way_two_panel_interaction_and_slopes_p005.pdf"),
    plot = combined_3way,
    width = 15,
    height = fig_height_3way,
    bg = "white"
  )
  
  readr::write_csv(
    plot_3way_wide,
    paste0(output_path,"Longitudinal/mixed_models_3way_panelA_formal_interaction_p005.csv")
  )
  
  readr::write_csv(
    plot_3way_long,
    paste0(output_path,"Longitudinal/mixed_models_3way_panelB_group_specific_slopes_p005.csv")
  )
}


# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
# Coefficient plots for mixed model results\n ----
# Top 20 proteins by nominal significance (raw p < 0.1)\n 
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
#"========================================================= \n
# Coefficient plots for mixed model results\n
# Top 20 proteins by nominal significance (raw p < 0.1)\n
# after excluding selected core biomarkers\n
#\n
# Panels:\n
#   A) baseline biomarker × time\n
#   B) group × baseline biomarker × time\n
#
# Figure style:\n
#   - beta coefficients on x-axis\n
#   - 95% CI from beta ± 1.96 * SE\n
#   - color by raw p-value category\n
#   - same symmetric x-axis in both panels\n
# ========================================================="

# .............................................................................
# File path
# .............................................................................
file_path <- paste0(output_path, "Longitudinal/ztotal_composite_change_mixed_model_baseline_bm_log_z_imputed_zeros_bmi_adj.csv")

# .............................................................................
# Proteins to exclude
# .............................................................................
exclude_proteins <- c(
  "pTau181",
  "pTau231",
  "pTau217",
  "AB38",
  "NEFL",
  "GFAP",
  "AB40", 
  "lumipulse_ptau217"
)

# .............................................................................
# Read data
# .............................................................................
res <- read_csv(file_path, show_col_types = FALSE)

# Rename first column if needed
if (!"Protein" %in% names(res)) {
  names(res)[1] <- "Protein"
}

# Clean protein names and exclude selected proteins
res <- res %>%
  mutate(
    Protein = str_remove(Protein, "^nulisa_")
  ) %>%
  filter(!Protein %in% exclude_proteins)

# .............................................................................
# Helper function to build term table
# .............................................................................
build_term_table <- function(data, beta_col, se_col, p_col, top_n = 25, p_cutoff = 0.1) {
  data %>%
    transmute(
      Protein,
      beta = .data[[beta_col]],
      se   = .data[[se_col]],
      p    = .data[[p_col]]
    ) %>%
    filter(!is.na(beta), !is.na(se), !is.na(p)) %>%
    mutate(
      ci_low  = beta - 1.96 * se,
      ci_high = beta + 1.96 * se,
      sig = case_when(
        p < 0.05 ~ "p < 0.05",
        p < p_cutoff ~ "0.05 <= p < 0.1",
        TRUE ~ "NS"
      ),
      abs_beta = abs(beta)
    ) %>%
    filter(p < p_cutoff) %>%
    arrange(p, desc(abs_beta)) %>%
    slice_head(n = top_n) %>%
    arrange(beta) %>%
    mutate(Protein = factor(Protein, levels = Protein))
}

# .............................................................................
# Build tables for the two interaction terms
# .............................................................................
plot_bmyear <- build_term_table(
  data = res,
  beta_col = "bm:year coeff",
  se_col   = "bm:year std",
  p_col    = "bm:year pvalue",
  top_n    = 25,
  p_cutoff = 0.1
)

plot_3way <- build_term_table(
  data = res,
  beta_col = "group:bm:year coeff",
  se_col   = "group:bm:year std",
  p_col    = "group:bm:year pvalue",
  top_n    = 25,
  p_cutoff = 0.1
)

# .............................................................................
# Shared symmetric x-axis limits
# .............................................................................
max_val <- max(
  abs(c(
    plot_bmyear$ci_low,
    plot_bmyear$ci_high,
    plot_3way$ci_low,
    plot_3way$ci_high
  )),
  na.rm = TRUE
)

# Add a small margin
max_val <- max_val * 1.08
x_limits <- c(-max_val, max_val)

# .............................................................................
# Plot function
# .............................................................................
make_coef_plot <- function(df, panel_label, title, subtitle) {
  ggplot(df, aes(x = beta, y = Protein, color = sig)) +
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
        "0.05 <= p < 0.1" = "#6baed6",
        "NS" = "grey70"
      ),
      drop = FALSE
    ) +
    coord_cartesian(xlim = x_limits) +
    labs(
      title = paste0(panel_label, "  ", title),
      subtitle = subtitle,
      x = expression(beta ~ "coefficient"),
      y = NULL,
      color = NULL
    ) +
    theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      plot.subtitle = element_text(size = 10),
      axis.text.y = element_text(size = 11),
      axis.text.x = element_text(size = 11),
      axis.title.x = element_text(size = 11),
      legend.position = "top",
      legend.text = element_text(size = 9)
    )
}

# .............................................................................
# Create plots
# .............................................................................
p1 <- make_coef_plot(
  df = plot_bmyear,
  panel_label = "A",
  title = "Baseline biomarker association with cognitive change",
  subtitle = paste0(
    "Top 25 proteins ranked by p value",
    "\nInteraction term: baseline biomarker × time"
  )
)

p2 <- make_coef_plot(
  df = plot_3way,
  panel_label = "B",
  title = paste0(test_group_val," modification of biomarker-associated cognitive change"),
  subtitle = paste0(
    "Top 25 proteins ranked by p value",
    "\nInteraction term: group × baseline biomarker × time"
  )
)
combined_plot <- p1 + p2 + plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "top")

print(combined_plot)

# .............................................................................
# Save figure
# .............................................................................
ggsave(
  filename = paste0(output_path,"Longitudinal/mixed models_top 25_ forest plots_imputed_wins_log_z_mitt_BMI_adj.png"),
  plot = combined_plot,
  width = 13,
  height = 9,
  dpi = 600
)

ggsave(
  filename = paste0(output_path,"Longitudinal/mixed models_top 25_ forest plots_imputed_wins_log_z_mitt_BMI_adj.svg"),
  plot = combined_plot,
  device = svglite::svglite,
  width = 13,
  height = 9
)
# .............................................................................
# Save plotted tables
# .............................................................................
write_csv(
  plot_bmyear,
  paste0(output_path,"Longitudinal/top25_bm_year_rawp_lt_0.1_excluding_selected_proteinsv5.csv")
)

write_csv(
  plot_3way,
  paste0(output_path,"Longitudinal/top25_group_bm_year_rawp_lt_0.1_excluding_selected_proteinsv5.csv")
)




# /////////////////////////////////////////////////////////////////////////////
# BM change vs intervention ------
# /////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////
## Mixed-effects models for all biomarkers, outcome: bm over time, predictor: group ------
# /////////////////////////////////////////////////////////////////////////////

# Mixed-effects models for all NULISA biomarkers
# + lumipulse_ptau217
# Outcome: biomarker over time
# Predictor of interest: intervention group
#
# Specifications:
# - random intercept : (1 | ID)
# - campid reference = Helsinki (101)
# - group reference = control 
# - sex reference = man (1)
# - winsorization at 1% and 99%, baseline and 2-year separately
# - log transformation AFTER winsorization
# - NO z-scaling
# - remove patients without baseline biomarker
# - remove patients without 2-year biomarker
# - include only patients with baseline cognition and at least one follow-up cognition
# - remove patients without one of the cofactors


# .............................................................................
# File path
# .............................................................................
file_path <- paste0(processed_path,"dataset_original.csv")  # was luovutettavat_85_1_merge.csv (same variables)


# .............................................................................
# Load data
# .............................................................................
df_all <- read_csv(file_path, show_col_types = FALSE)


df_all <- df_all %>%
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
  var_group,
  "age_bl1",
  "sex",
  "educ_years_BL1",
  "campid",
  "ztotal_composite_BL2",
  "ztotal_composite_12kk",
  "ztotal_composite_24kk"
)


# .............................................................................
# One-biomarker model runner
# .............................................................................
run_one_biomarker <- function(bm_base, data) {
  
  bm0 <- paste0(bm_base, "_0")
  bm2 <- paste0(bm_base, "_2")
  
  df_model <- data %>%
    select(all_of(c(base_vars, bm0, bm2))) %>%
    filter(
      !is.na(.data[[bm0]]),
      !is.na(.data[[bm2]]),
      !is.na(ztotal_composite_BL2),
      (!is.na(ztotal_composite_12kk) | !is.na(ztotal_composite_24kk)),
      !is.na(age_bl1),
      !is.na(sex),
      !is.na(educ_years_BL1),
      !is.na(campid),
      !is.na(.data[[var_group]])
    ) %>%
    mutate(
      bm0_w = winsorize(.data[[bm0]], p = 0.01),
      bm2_w = winsorize(.data[[bm2]], p = 0.01)
    )
  
  # log requires positive values
  if ((needs_log(bm_base) && any(df_model$bm0_w <= 0, na.rm = TRUE)) || (needs_log(bm_base) && any(df_model$bm2_w <= 0, na.rm = TRUE))) {
    return(tibble(
      biomarker = str_remove(bm_base, "^nulisa_"),
      n_ids = n_distinct(df_model$ID),
      n_obs = NA_integer_,
      error = "Non-positive values after winsorization; log not possible"
    ))
  }
  
  df_model <- df_model %>%
    mutate(
      bm0_log = transform_bm(bm0_w, bm_base),
      bm2_log = transform_bm(bm2_w, bm_base)
    )
  
  df_long <- df_model %>%
    transmute(
      ID,
      !!rlang::sym(var_group),
      age_bl1,
      sex,
      educ_years_BL1,
      campid,
      biomarker_0 = bm0_log,
      biomarker_2 = bm2_log
    ) %>%
    pivot_longer(
      cols = c(biomarker_0, biomarker_2),
      names_to = "timepoint",
      values_to = "biomarker_value"
    ) 
  df_long <-df_long %>%
    mutate(
      year = case_when(
        timepoint == "biomarker_0" ~ 0,
        timepoint == "biomarker_2" ~ 2
      ),      
      !!rlang::sym(var_group)  := factor(df_long[[var_group]]),
      sex = factor(sex),
      campid = factor(campid)
    ) %>%
    filter(!is.na(biomarker_value))
  
  # Reference levels

  if ("3" %in% levels(df_long$sub_sample_id)) {
    df_long$sub_sample_id <- relevel(df_long$sub_sample_id, ref = "3")
  }
    
  if ("1" %in% levels(df_long$sex)) {
    df_long$sex <- relevel(df_long$sex, ref = "1")
  }
  if ("101" %in% levels(df_long$campid)) {
    df_long$campid <- relevel(df_long$campid, ref = "101")
  }
  
  model <- tryCatch(
    lmer(
      paste0(
        "biomarker_value ~ year * ", var_group,
        " + age_bl1 + sex + educ_years_BL1 + campid +",
        " (1 | ID)"
      ),

      data = df_long,
      REML = FALSE
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
    
    `group coeff` = get_term_value(fixed, ref_group, estimate),
    `group std` = get_term_value(fixed, ref_group, std.error),
    `group pvalue` = get_term_value(fixed, ref_group, p.value),
    
    `year coeff` = get_term_value(fixed, "year", estimate),
    `year std` = get_term_value(fixed, "year", std.error),
    `year pvalue` = get_term_value(fixed, "year", p.value),
    
    `group:year coeff` = get_term_value(fixed, paste0("year:",ref_group), estimate),
    `group:year std` = get_term_value(fixed, paste0("year:",ref_group), std.error),
    `group:year pvalue` = get_term_value(fixed, paste0("year:",ref_group), p.value),
    
    `age_0y coeff` = get_term_value(fixed, "age_bl1", estimate),
    `age_0y std` = get_term_value(fixed, "age_bl1", std.error),
    `age_0y pvalue` = get_term_value(fixed, "age_bl1", p.value),
    
    `sex coeff` = get_term_value(fixed, "sex2", estimate),
    `sex std` = get_term_value(fixed, "sex2", std.error),
    `sex pvalue` = get_term_value(fixed, "sex2", p.value),
    
    `educ_years_0y coeff` = get_term_value(fixed, "educ_years_BL1", estimate),
    `educ_years_0y std` = get_term_value(fixed, "educ_years_BL1", std.error),
    `educ_years_0y pvalue` = get_term_value(fixed, "educ_years_BL1", p.value),
    
    error = NA_character_
  )
}

# .............................................................................
# Run all models
# .............................................................................
results_all <- map_dfr(biomarker_names, run_one_biomarker, data = df_all, .progress = interactive())

# .............................................................................
# Add FDR columns
# .............................................................................
p_cols <- grep("pvalue$", names(results_all), value = TRUE)

for (pcol in p_cols) {
  fdr_col <- paste0(pcol, " FDR")
  results_all[[fdr_col]] <- p.adjust(results_all[[pcol]], method = "fdr")
}

# .............................................................................
# Round numeric columns
# .............................................................................
results_all <- results_all %>%
  mutate(across(where(is.numeric), ~ round(.x, 6)))

# .............................................................................
# Save results
# .............................................................................
write_csv(
  results_all,
  paste0(output_path,"Longitudinal/mixed_model_all_biomarkers_outcome_group_predictor_log_noz_atleast1FUcognition.csv")
)




# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
## Linear models for all biomarkers -------------------------------------------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////

# Linear models for all NULISA biomarkers + lumipulse_ptau217
# Outcome: biomarker change from baseline to 2 years
# Predictor of interest: interv. group
#
# Specifications:
# - outcome = biomarker_2y_log - biomarker_0_log
# - adjust for baseline biomarker level
# - adjust for age_bl1, sex, educ_years_BL1, bmi_bl2, campid
# - campid reference = Helsinki (101)
# - group reference = control
# - sex reference = man (1)
# - winsorization at 1% and 99%, baseline and 2-year separately
# - log transformation AFTER winsorization
# - NO z-scaling
# - remove patients without baseline biomarker
# - remove patients without 2-year biomarker
# - remove patients without one of the cofactors


# .............................................................................
# File path
# .............................................................................
file_path <- paste0(processed_path,"dataset_original.csv")
# .............................................................................
# Load data
# .............................................................................
df_all <- read_csv(file_path, show_col_types = FALSE)

df_all <- df_all %>%
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
  var_group,
  "age_bl1",
  "sex",
  "educ_years_BL1",
  "bmi_bl2",
  "campid"
)



# .............................................................................
# One-biomarker model runner
# .............................................................................
run_one_biomarker <- function(bm_base, data) {
  
  bm0 <- paste0(bm_base, "_0")
  bm2 <- paste0(bm_base, "_2")

  
  df_model <- data %>%
    select(all_of(c(base_vars, bm0, bm2))) %>%
    filter(
      !is.na(.data[[bm0]]),
      !is.na(.data[[bm2]]),
      !is.na(age_bl1),
      !is.na(sex),
      !is.na(educ_years_BL1),
      !is.na(campid),
      !is.na(bmi_bl2),
      !is.na(.data[[var_group]])
    ) %>%
    mutate(
      bm0_w = winsorize(.data[[bm0]], p = 0.01),
      bm2_w = winsorize(.data[[bm2]], p = 0.01)
    )
  
  # log requires positive values
  if ((needs_log(bm_base) && any(df_model$bm0_w <= 0, na.rm = TRUE)) ||
      (needs_log(bm_base) && any(df_model$bm2_w <= 0, na.rm = TRUE))) {
    return(tibble(
      biomarker = str_remove(bm_base, "^nulisa_"),
      n = nrow(df_model),
      
      `Intercept coeff` = NA_real_,
      `Intercept std` = NA_real_,
      `Intercept pvalue` = NA_real_,
      
      `group coeff` = NA_real_,
      `group std` = NA_real_,
      `group pvalue` = NA_real_,
      
      `baseline coeff` = NA_real_,
      `baseline std` = NA_real_,
      `baseline pvalue` = NA_real_,
      
      `age_0y coeff` = NA_real_,
      `age_0y std` = NA_real_,
      `age_0y pvalue` = NA_real_,
      
      `sex coeff` = NA_real_,
      `sex std` = NA_real_,
      `sex pvalue` = NA_real_,
      
      `educ_years_0y coeff` = NA_real_,
      `educ_years_0y std` = NA_real_,
      `educ_years_0y pvalue` = NA_real_,
      
      `bmi_0y coeff` = NA_real_,
      `bmi_0y std` = NA_real_,
      `bmi_0y pvalue` = NA_real_,
      
      error = "Non-positive values after winsorization; log not possible"
    ))
  }
  
  df_model <- df_model %>%
    mutate(
      bm0_log = transform_bm(bm0_w, bm_base),
      bm2_log = transform_bm(bm2_w, bm_base),
      bm_change = bm2_log - bm0_log,
      !!rlang::sym(var_group) := relevel(factor(df_model[[var_group]]), ref = ref_control),
      sex = relevel(factor(sex), ref = "1"),
      campid = relevel(factor(campid), ref = "101")
    )
  
  model <- tryCatch(
    lm(
      paste0("bm_change ~ bm0_log + ",var_group," + age_bl1 + sex + educ_years_BL1 + bmi_bl2 + campid"),
      data = df_model
    ),
    error = function(e) e
  )
  
  if (inherits(model, "error")) {
    return(tibble(
      biomarker = str_remove(bm_base, "^nulisa_"),
      n = nrow(df_model),
      
      `Intercept coeff` = NA_real_,
      `Intercept std` = NA_real_,
      `Intercept pvalue` = NA_real_,
      
      `group coeff` = NA_real_,
      `group std` = NA_real_,
      `group pvalue` = NA_real_,
      
      `baseline coeff` = NA_real_,
      `baseline std` = NA_real_,
      `baseline pvalue` = NA_real_,
      
      `age_0y coeff` = NA_real_,
      `age_0y std` = NA_real_,
      `age_0y pvalue` = NA_real_,
      
      `sex coeff` = NA_real_,
      `sex std` = NA_real_,
      `sex pvalue` = NA_real_,
      
      `educ_years_0y coeff` = NA_real_,
      `educ_years_0y std` = NA_real_,
      `educ_years_0y pvalue` = NA_real_,
      
      `bmi_0y coeff` = NA_real_,
      `bmi_0y std` = NA_real_,
      `bmi_0y pvalue` = NA_real_,
      
      error = model$message
    ))
  }
  
  fixed <- tidy(model) %>%
    select(term, estimate, std.error, p.value)
  
  tibble(
    biomarker = str_remove(bm_base, "^nulisa_"),
    n = nrow(df_model),
    
    `Intercept coeff` = get_term_value(fixed, "(Intercept)", estimate),
    `Intercept std` = get_term_value(fixed, "(Intercept)", std.error),
    `Intercept pvalue` = get_term_value(fixed, "(Intercept)", p.value),
    
    `group coeff` = get_term_value(fixed, ref_group, estimate),
    `group std` = get_term_value(fixed, ref_group, std.error),
    `group pvalue` = get_term_value(fixed, ref_group, p.value),
    
    `baseline coeff` = get_term_value(fixed, "bm0_log", estimate),
    `baseline std` = get_term_value(fixed, "bm0_log", std.error),
    `baseline pvalue` = get_term_value(fixed, "bm0_log", p.value),
    
    `age_0y coeff` = get_term_value(fixed, "age_bl1", estimate),
    `age_0y std` = get_term_value(fixed, "age_bl1", std.error),
    `age_0y pvalue` = get_term_value(fixed, "age_bl1", p.value),
    
    `sex coeff` = get_term_value(fixed, "sex2", estimate),
    `sex std` = get_term_value(fixed, "sex2", std.error),
    `sex pvalue` = get_term_value(fixed, "sex2", p.value),
    
    `educ_years_0y coeff` = get_term_value(fixed, "educ_years_BL1", estimate),
    `educ_years_0y std` = get_term_value(fixed, "educ_years_BL1", std.error),
    `educ_years_0y pvalue` = get_term_value(fixed, "educ_years_BL1", p.value),
    
    `bmi_0y coeff` = get_term_value(fixed, "bmi_bl2", estimate),
    `bmi_0y std` = get_term_value(fixed, "bmi_bl2", std.error),
    `bmi_0y pvalue` = get_term_value(fixed, "bmi_bl2", p.value),
    
    error = NA_character_
  )
}
# .............................................................................
# Run all models
# .............................................................................
results_all <- map_dfr(biomarker_names, run_one_biomarker, data = df_all, .progress = interactive())

# .............................................................................
# Add FDR columns
# .............................................................................
p_cols <- grep("pvalue$", names(results_all), value = TRUE)

for (pcol in p_cols) {
  fdr_col <- paste0(pcol, " FDR")
  results_all[[fdr_col]] <- p.adjust(results_all[[pcol]], method = "fdr")
}

# .............................................................................
# Round numeric columns
# .............................................................................
results_all <- results_all %>%
  mutate(across(where(is.numeric), ~ round(.x, 6)))

# .............................................................................
# Save results
# .............................................................................
write_csv(
  results_all,
  paste0(output_path, "Longitudinal/linear_model_biomarker_change_adjusted_for_baseline_log_noz_imputed_bmi_adj_mitt.csv")
)

# .............................................................................
# Top bm 
# .............................................................................

top25_tables <- create_top25_tables(
  results = results_all,
  output_path = output_path,
  folder = "Longitudinal",
  method = "bm_change_linear",
  p = "group pvalue"
)

top25_df <- top25_tables$top25_df
top25_df_low <- top25_tables$top25_df_low



# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
## Volcano plot for group effect on biomarker change--------------------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////

# Volcano plot for group effect on biomarker change
# Uses results_all from previous script
# x-axis  = group coeff
# y-axis  = -log10(group pvalue)
# Labels  = top 15 proteins by p-value
# Excludes predefined AD biomarkers


origin_model <- "linear_model" # "mixed_model", "linear_model"

if (origin_model == "mixed_model"){
  results_all <- read_csv(paste0(output_path,"Longitudinal/mixed_model_all_biomarkers_outcome_group_predictor_log_noz_atleast1FUcognition_mitt.csv"), show_col_types = FALSE)
}else{
  results_all <- read_csv(paste0(output_path,"Longitudinal/linear_model_biomarker_change_adjusted_for_baseline_log_noz_imputed_bmi_adj_mitt.csv"), show_col_types = FALSE)
}
# .............................................................................
# Proteins to exclude
# .............................................................................
exclude_proteins <- c(
  "pTau181",
  "pTau231",
  "pTau217",
  "AB38",
  "NEFL",
  "GFAP",
  "AB40","AB42",
  "lumipulse_ptau217"
)

# .............................................................................
# Prepare volcano data
# .............................................................................

if (origin_model == "mixed_model"){
  volcano_df <- results_all %>%
    filter(
      !biomarker %in% exclude_proteins,
      !is.na(`group:year coeff`),
      !is.na(`group:year pvalue`),
      `group pvalue` > 0
    ) %>%
    mutate(
      neglog10p = -log10(`group:year pvalue`),
      status = case_when(
        `group:year pvalue` < 0.05 & `group:year coeff` > 0 ~ "Positive, p < 0.05",
        `group:year pvalue` < 0.05 & `group:year coeff` < 0 ~ "Negative, p < 0.05",
        `group:year pvalue` < 0.10 & `group:year coeff` > 0 ~ "Positive, 0.05 <= p < 0.10",
        `group:year pvalue` < 0.10 & `group:year coeff` < 0 ~ "Negative, 0.05 <= p < 0.10",
        TRUE ~ "Not significant"
      )
    )
  # .............................................................................
  # Labels = top 15 by p-value
  # .............................................................................
  label_df <- volcano_df %>%
    arrange(`group:year pvalue`, desc(abs(`group:year coeff`))) %>%
    slice_head(n = 15)
}else{
  volcano_df <- results_all %>%
    filter(
      !biomarker %in% exclude_proteins,
      !is.na(`group coeff`),
      !is.na(`group pvalue`),
      `group pvalue` > 0
    ) %>%
    mutate(
      neglog10p = -log10(`group pvalue`),
      status = case_when(
        `group pvalue` < 0.05 & `group coeff` > 0 ~ "Positive, p < 0.05",
        `group pvalue` < 0.05 & `group coeff` < 0 ~ "Negative, p < 0.05",
        `group pvalue` < 0.10 & `group coeff` > 0 ~ "Positive, 0.05 <= p < 0.10",
        `group pvalue` < 0.10 & `group coeff` < 0 ~ "Negative, 0.05 <= p < 0.10",
        TRUE ~ "Not significant"
      )
    )
  # .............................................................................
  # Labels = top 15 by p-value
  # .............................................................................
  label_df <- volcano_df %>%
    arrange(`group pvalue`, desc(abs(`group coeff`))) %>%
    slice_head(n = 15)
  
}




# .............................................................................
# Colors
# .............................................................................
volcano_colors <- c(
  "Negative, p < 0.05" = "#CC6677",
  "Negative, 0.05 <= p < 0.10" = "#E7A1B0",
  "Not significant" = "grey70",
  "Positive, 0.05 <= p < 0.10" = "#88CCAA",
  "Positive, p < 0.05" = "#228833"
)

# .............................................................................
# Plot
# .............................................................................
if (origin_model == "mixed_model"){
  
  p_volcano <- ggplot(
    volcano_df,
    aes(x = `group:year coeff`, y = neglog10p, color = status)
  ) +
    geom_hline(
      yintercept = -log10(0.05),
      linetype = "dashed",
      linewidth = 0.5,
      color = "grey45"
    ) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.5,
      color = "grey55"
    ) +
    geom_point(size = 2.8, alpha = 0.9) +
    geom_text_repel(
      data = label_df,
      aes(label = biomarker),
      size = 3.4,
      box.padding = 0.3,
      point.padding = 0.2,
      segment.color = "grey60",
      max.overlaps = Inf,
      show.legend = FALSE
    ) +
    scale_color_manual(values = volcano_colors) +
    labs(
      title = paste0(test_group_val," effect on biomarker change"),
      x = paste0("Beta coefficient for ",test_group_val," group"),
      y = expression(-log[10] * "(raw p-value)"),
      color = NULL
    ) +
    theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      plot.subtitle = element_text(size = 10),
      legend.position = "right",
      legend.text = element_text(size = 9)
    )
}else{
  
  p_volcano <- ggplot(
    volcano_df,
    aes(x = `group coeff`, y = neglog10p, color = status)
  ) +
    geom_hline(
      yintercept = -log10(0.05),
      linetype = "dashed",
      linewidth = 0.5,
      color = "grey45"
    ) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.5,
      color = "grey55"
    ) +
    geom_point(size = 2.8, alpha = 0.9) +
    geom_text_repel(
      data = label_df,
      aes(label = biomarker),
      size = 3.4,
      box.padding = 0.3,
      point.padding = 0.2,
      segment.color = "grey60",
      max.overlaps = Inf,
      show.legend = FALSE
    ) +
    scale_color_manual(values = volcano_colors) +
    labs(
      title = paste0(test_group_val," effect on biomarker change"),
      x = paste0("Beta coefficient for ",test_group_val," group"),
      y = expression(-log[10] * "(raw p-value)"),
      color = NULL
    ) +
    theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      plot.subtitle = element_text(size = 11),
      legend.position = "right",
      legend.text = element_text(size = 10)
    )
}

print(p_volcano)

# .............................................................................
# Save outputs
# .............................................................................
ggsave(
  filename = paste0(output_path,"Longitudinal/volcano_from_",origin_model,"_intervention_effect_biomarker_change_imputed_BMI_adj_mitt.png"),
  plot = p_volcano,
  width = 9,
  height = 7,
  dpi = 600
)

ggsave(
  filename = paste0(output_path,"Longitudinal/volcano_from_",origin_model,"_intervention_effect_biomarker_change_imputed_BMI_adj_mitt.svg"),
  plot = p_volcano,
  device = svglite::svglite,
  width = 9,
  height = 7
)


write_csv(
  volcano_df,
  paste0(output_path,"Longitudinal/volcano_from_",origin_model,"_intervention_effect_biomarker_change_table_imputed_mitt.csv")
)



#summary(df_all$nulisa_CHIT1_0)
#summary(df_all$nulisa_GDF15_0)

# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
## Adjusted biomarker change by intervention class for all biomarkers ------------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////

# Adjusted biomarker change by intervention class for all biomarkers
# Outcome: biomarker change from baseline to 2 years
# Model: change ~ baseline biomarker + group + age + sex + education + site +BMI
#
# Output per biomarker:
# - adjusted mean change in control
# - adjusted mean change in intervention groupo
# - intervention group - control difference
# - p-value for group difference

# .............................................................................
# File path
# .............................................................................
file_path <- paste0(processed_path,"dataset_original.csv")

# .............................................................................
# Load data
# .............................................................................
df_all <- read_csv(file_path, show_col_types = FALSE)

df_all <- df_all %>%
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
  var_group,
  "age_bl1",
  "sex",
  "educ_years_BL1",
  "bmi_bl2",
  "campid"
)

# .............................................................................
# One-biomarker model runner
# .............................................................................
run_one_biomarker <- function(bm_base, data) {
  
  bm0 <- paste0(bm_base, "_0")
  bm2 <- paste0(bm_base, "_2")

  
  df_model <- data %>%
    select(all_of(c(base_vars, bm0, bm2))) %>%
    filter(
      !is.na(.data[[bm0]]),
      !is.na(.data[[bm2]]),
      !is.na(age_bl1),
      !is.na(sex),
      !is.na(educ_years_BL1),
      !is.na(campid),
      !is.na(bmi_bl2),
      !is.na(.data[[var_group]])
    ) %>%
    mutate(
      bm0_w = winsorize(.data[[bm0]], p = 0.01),
      bm2_w = winsorize(.data[[bm2]], p = 0.01)
    )
  
  # log requires positive values
  if ((needs_log(bm_base) && any(df_model$bm0_w <= 0, na.rm = TRUE)) ||
      (needs_log(bm_base) && any(df_model$bm2_w <= 0, na.rm = TRUE))) {
    return(tibble(
      biomarker = str_remove(bm_base, "^nulisa_"),
      n = nrow(df_model),
      control_change = NA_real_,
      intervention_change = NA_real_,
      group_difference = NA_real_,
      group_difference_se = NA_real_,
      group_pvalue = NA_real_,
      error = "Non-positive values after winsorization; log not possible"
    ))
  }
  
  df_model <- df_model %>%
    mutate(
      bm0_log = transform_bm(bm0_w, bm_base),
      bm2_log = transform_bm(bm2_w, bm_base),
      bm_change = bm2_log - bm0_log,
      !!rlang::sym(var_group) := relevel(factor(df_model[[var_group]]), ref = ref_control),
      sex = relevel(factor(sex), ref = "1"),
      campid = relevel(factor(campid), ref = "101")
    )
  
  model <- tryCatch(
    lm(
      paste0("bm_change ~ bm0_log + ",var_group," + age_bl1 + sex + educ_years_BL1 + bmi_bl2 + campid"),
      data = df_model
    ),
    error = function(e) e
  )
  
  if (inherits(model, "error")) {
    return(tibble(
      biomarker = str_remove(bm_base, "^nulisa_"),
      n = nrow(df_model),
      control_change = NA_real_,
      intervention_change = NA_real_,
      group_difference = NA_real_,
      group_difference_se = NA_real_,
      group_pvalue = NA_real_,
      error = model$message
    ))
  }
  emm <- tryCatch(
    emmeans(model, ~ sub_sample_id),
    error = function(e) e
  )
  
  if (inherits(emm, "error")) {
    return(tibble(
      biomarker = str_remove(bm_base, "^nulisa_"),
      n = nrow(df_model),
      control_change = NA_real_,
      intervention_change = NA_real_,
      group_difference = NA_real_,
      group_difference_se = NA_real_,
      group_pvalue = NA_real_,
      error = emm$message
    ))
  }
  
  emm_df <- as.data.frame(emm)
  contrast_df <- as.data.frame(contrast(emm, method = "revpairwise"))
  
  control_est <- emm_df %>%
    filter(emm_df[[var_group]] %in% c("control", "Control", 3)) %>%
    pull(emmean)
  if (length(control_est) == 0) control_est <- NA_real_
  
  intervention_est <- emm_df %>%
    filter(emm_df[[var_group]] %in% c("high", "intervention", "Intervention", 2)) %>%
    pull(emmean)
  if (length(intervention_est) == 0) intervention_est <- NA_real_
  
  tibble(
    biomarker = str_remove(bm_base, "^nulisa_"),
    n = nrow(df_model),
    control_change = control_est,
    intervention_change = intervention_est,
    group_difference = ifelse(nrow(contrast_df) > 0, contrast_df$estimate[1], NA_real_),
    group_difference_se = ifelse(nrow(contrast_df) > 0, contrast_df$SE[1], NA_real_),
    group_pvalue = ifelse(nrow(contrast_df) > 0, contrast_df$p.value[1], NA_real_),
    error = NA_character_
  )
}

# .............................................................................
# Run all biomarkers
# .............................................................................
results_groups <- map_dfr(biomarker_names, run_one_biomarker, data = df_all, .progress = interactive())

# .............................................................................
# FDR correction
# .............................................................................
results_groups <- results_groups %>%
  mutate(
    group_pvalue_FDR = p.adjust(group_pvalue, method = "fdr")
  )

# .............................................................................
# Add direction labels
# .............................................................................
results_groups <- results_groups %>%
  mutate(
    control_direction = case_when(
      is.na(control_change) ~ NA_character_,
      control_change > 0 ~ "Increase",
      control_change < 0 ~ "Decrease",
      TRUE ~ "No change"
    ),
    intervention_direction = case_when(
      is.na(intervention_change) ~ NA_character_,
      intervention_change > 0 ~ "Increase",
      intervention_change < 0 ~ "Decrease",
      TRUE ~ "No change"
    )
  ) %>%
  mutate(across(where(is.numeric), ~ round(.x, 6)))

# .............................................................................
# Save
# .............................................................................
write_csv(
  results_groups,
  paste0(output_path,"Longitudinal/adjusted_group_means_biomarker_change_bmi_adj_mitt.csv")
)



# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
# APOE-stratified bm ~ year * group ----------------------------------------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////

# /////////////////////////////////////////////////////////////////////////////
## Linear model ---------------------
# /////////////////////////////////////////////////////////////////////////////

# .............................................................................
# File path
# .............................................................................
file_path <- paste0(processed_path,"dataset_original.csv")
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
  var_group,
  "apoe4_status",
  "age_bl1",
  "sex",
  "educ_years_BL1",
  "bmi_bl2",
  "campid"
)

# .............................................................................
# One-biomarker model runner within one APOE stratum
# .............................................................................
run_one_biomarker_stratum <- function(bm_base, data, apoe_level) {

  bm0 <- paste0(bm_base, "_0")
  bm2 <- paste0(bm_base, "_2")

  
  df_model <- data %>%
    filter(apoe4_status == apoe_level) %>%
    select(all_of(c(base_vars, bm0, bm2))) %>%
    filter(
      !is.na(.data[[bm0]]),
      !is.na(.data[[bm2]]),
      !is.na(age_bl1),
      !is.na(sex),
      !is.na(educ_years_BL1),
      !is.na(campid),
      !is.na(bmi_bl2),
      !is.na(.data[[var_group]]),
      !is.na(apoe4_status)
    ) %>%
    mutate(
      bm0_w = winsorize(.data[[bm0]], p = 0.01),
      bm2_w = winsorize(.data[[bm2]], p = 0.01)
    )
  
  
  
  
  if (n_distinct(df_model$ID) < 20) {
    return(tibble(
      apoe4_status = as.character(apoe_level),
      biomarker = str_remove(bm_base, "^nulisa_"),
      n_ids = n_distinct(df_model$ID),
      n_obs = NA_integer_,
      error = "Too few participants in APOE stratum"
    ))
  }
  # log requires positive values
  if ((needs_log(bm_base) && any(df_model$bm0_w <= 0, na.rm = TRUE)) ||
      (needs_log(bm_base) && any(df_model$bm2_w <= 0, na.rm = TRUE))) {
    return(tibble(
      biomarker = str_remove(bm_base, "^nulisa_"),
      n = nrow(df_model),
      
      `Intercept coeff` = NA_real_,
      `Intercept std` = NA_real_,
      `Intercept pvalue` = NA_real_,
      
      `group coeff` = NA_real_,
      `group std` = NA_real_,
      `group pvalue` = NA_real_,
      
      `baseline coeff` = NA_real_,
      `baseline std` = NA_real_,
      `baseline pvalue` = NA_real_,
      
      `age_0y coeff` = NA_real_,
      `age_0y std` = NA_real_,
      `age_0y pvalue` = NA_real_,
      
      `sex coeff` = NA_real_,
      `sex std` = NA_real_,
      `sex pvalue` = NA_real_,
      
      `educ_years_0y coeff` = NA_real_,
      `educ_years_0y std` = NA_real_,
      `educ_years_0y pvalue` = NA_real_,
      
      `bmi_0y coeff` = NA_real_,
      `bmi_0y std` = NA_real_,
      `bmi_0y pvalue` = NA_real_,
      
      error = "Non-positive values after winsorization; log not possible"
    ))
  }
  
  
  
  df_model <- df_model %>%
    mutate(
      bm0_log = transform_bm(bm0_w, bm_base),
      bm2_log = transform_bm(bm2_w, bm_base),
      bm_change = bm2_log - bm0_log,
      !!rlang::sym(var_group) := relevel(factor(df_model[[var_group]]), ref = ref_control),
      sex = relevel(factor(sex), ref = "1"),
      campid = relevel(factor(campid), ref = "101")
    )
  
  model <- tryCatch(
    lm(
      paste0("bm_change ~ bm0_log + ",var_group," + age_bl1 + sex + educ_years_BL1 + bmi_bl2 + campid"),
      data = df_model
    ),
    error = function(e) e
  )
  
  if (inherits(model, "error")) {
    return(tibble(
      biomarker = str_remove(bm_base, "^nulisa_"),
      n = nrow(df_model),
      
      `Intercept coeff` = NA_real_,
      `Intercept std` = NA_real_,
      `Intercept pvalue` = NA_real_,
      
      `group coeff` = NA_real_,
      `group std` = NA_real_,
      `group pvalue` = NA_real_,
      
      `baseline coeff` = NA_real_,
      `baseline std` = NA_real_,
      `baseline pvalue` = NA_real_,
      
      `age_0y coeff` = NA_real_,
      `age_0y std` = NA_real_,
      `age_0y pvalue` = NA_real_,
      
      `sex coeff` = NA_real_,
      `sex std` = NA_real_,
      `sex pvalue` = NA_real_,
      
      `educ_years_0y coeff` = NA_real_,
      `educ_years_0y std` = NA_real_,
      `educ_years_0y pvalue` = NA_real_,
      
      `bmi_0y coeff` = NA_real_,
      `bmi_0y std` = NA_real_,
      `bmi_0y pvalue` = NA_real_,
      
      error = model$message
    ))
  }
  
  fixed <- tidy(model) %>%
    select(term, estimate, std.error, p.value)
  
  tibble(
    apoe4_status = as.character(apoe_level),
    biomarker = str_remove(bm_base, "^nulisa_"),
    n = nrow(df_model),
    
    `Intercept coeff` = get_term_value(fixed, "(Intercept)", estimate),
    `Intercept std` = get_term_value(fixed, "(Intercept)", std.error),
    `Intercept pvalue` = get_term_value(fixed, "(Intercept)", p.value),
    
    `group coeff` = get_term_value(fixed, ref_group, estimate),
    `group std` = get_term_value(fixed, ref_group, std.error),
    `group pvalue` = get_term_value(fixed, ref_group, p.value),
    
    `baseline coeff` = get_term_value(fixed, "bm0_log", estimate),
    `baseline std` = get_term_value(fixed, "bm0_log", std.error),
    `baseline pvalue` = get_term_value(fixed, "bm0_log", p.value),
    
    `age_0y coeff` = get_term_value(fixed, "age_bl1", estimate),
    `age_0y std` = get_term_value(fixed, "age_bl1", std.error),
    `age_0y pvalue` = get_term_value(fixed, "age_bl1", p.value),
    
    `sex coeff` = get_term_value(fixed, "sex2", estimate),
    `sex std` = get_term_value(fixed, "sex2", std.error),
    `sex pvalue` = get_term_value(fixed, "sex2", p.value),
    
    `educ_years_0y coeff` = get_term_value(fixed, "educ_years_BL1", estimate),
    `educ_years_0y std` = get_term_value(fixed, "educ_years_BL1", std.error),
    `educ_years_0y pvalue` = get_term_value(fixed, "educ_years_BL1", p.value),
    
    `bmi_0y coeff` = get_term_value(fixed, "bmi_bl2", estimate),
    `bmi_0y std` = get_term_value(fixed, "bmi_bl2", std.error),
    `bmi_0y pvalue` = get_term_value(fixed, "bmi_bl2", p.value),
    
    error = NA_character_
  )
  
}

# .............................................................................
# Run all biomarkers in each APOE stratum
# .............................................................................
apoe_levels <- sort(unique(df_all$apoe4_status[!is.na(df_all$apoe4_status)]))

results_all <- map_dfr(
  apoe_levels,
  function(apoe_level) {
    map_dfr(
      biomarker_names,
      run_one_biomarker_stratum,
      data = df_all,
      apoe_level = apoe_level, 
      .progress = interactive()
    )
  }
)

# .............................................................................
# Add FDR within each APOE stratum
# .............................................................................
p_cols <- grep("pvalue$", names(results_all), value = TRUE)

results_all <- results_all %>%
  group_by(apoe4_status) %>%
  mutate(
    across(
      all_of(p_cols),
      ~ p.adjust(.x, method = "fdr"),
      .names = "{.col} FDR"
    )
  ) %>%
  ungroup()

# .............................................................................
# Round numeric columns
# .............................................................................
results_all <- results_all %>%
  mutate(across(where(is.numeric), ~ round(.x, 6)))

# .............................................................................
# Save results
# .............................................................................
write_csv(
  results_all,
  paste0(output_path,"Stratified/apoe_stratified_bm_change_linear_model_log_z_bmi_adj.csv")
)


# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
## Volcano plot for group effect on biomarker change--------------------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////

# Volcano plot for group effect on biomarker change
# Uses results_all from previous script
# x-axis  = group coeff
# y-axis  = -log10(group pvalue)
# Labels  = top 15 proteins by p-value
# Excludes predefined AD biomarkers


origin_model <- "linear_model" # "mixed_model", "linear_model"

results_all <- read_csv(paste0(output_path,"Stratified/apoe_stratified_bm_change_",origin_model,"_log_z_bmi_adj.csv"), show_col_types = FALSE)

# .............................................................................
# Proteins to exclude
# .............................................................................
exclude_proteins <- c(
  "pTau181",
  "pTau231",
  "pTau217",
  "AB38",
  "NEFL",
  "GFAP",
  "AB40","AB42",
  "lumipulse_ptau217"
)

for (apoe_grp in c(0,1)){
  # .............................................................................
  # Prepare volcano data
  # .............................................................................
  if (apoe_grp == 0){
    apoe_name = "APOE4 non-carriers"
  }else{
    apoe_name = "APOE4 carriers"
    
  }
  results_apoe <- results_all %>%
    filter(
      `apoe4_status` == apoe_grp
    )
  
  if (origin_model == "mixed_model"){
    volcano_df <- results_apoe %>%
      filter(
        !biomarker %in% exclude_proteins,
        !is.na(`group:year coeff`),
        !is.na(`group:year pvalue`),
        `group pvalue` > 0
      ) %>%
      mutate(
        neglog10p = -log10(`group:year pvalue`),
        status = case_when(
          `group:year pvalue` < 0.05 & `group:year coeff` > 0 ~ "Positive, p < 0.05",
          `group:year pvalue` < 0.05 & `group:year coeff` < 0 ~ "Negative, p < 0.05",
          `group:year pvalue` < 0.10 & `group:year coeff` > 0 ~ "Positive, 0.05 <= p < 0.10",
          `group:year pvalue` < 0.10 & `group:year coeff` < 0 ~ "Negative, 0.05 <= p < 0.10",
          TRUE ~ "Not significant"
        )
      )
    # .............................................................................
    # Labels = top 15 by p-value
    # .............................................................................
    label_df <- volcano_df %>%
      arrange(`group:year pvalue`, desc(abs(`group:year coeff`))) %>%
      slice_head(n = 15)
  }else{
    volcano_df <- results_apoe %>%
      filter(
        !biomarker %in% exclude_proteins,
        !is.na(`group coeff`),
        !is.na(`group pvalue`),
        `group pvalue` > 0
      ) %>%
      mutate(
        neglog10p = -log10(`group pvalue`),
        status = case_when(
          `group pvalue` < 0.05 & `group coeff` > 0 ~ "Positive, p < 0.05",
          `group pvalue` < 0.05 & `group coeff` < 0 ~ "Negative, p < 0.05",
          `group pvalue` < 0.10 & `group coeff` > 0 ~ "Positive, 0.05 <= p < 0.10",
          `group pvalue` < 0.10 & `group coeff` < 0 ~ "Negative, 0.05 <= p < 0.10",
          TRUE ~ "Not significant"
        )
      )
    # .............................................................................
    # Labels = top 15 by p-value
    # .............................................................................
    label_df <- volcano_df %>%
      arrange(`group pvalue`, desc(abs(`group coeff`))) %>%
      slice_head(n = 15)
    
  }
  
  
  
  
  # .............................................................................
  # Colors
  # .............................................................................
  volcano_colors <- c(
    "Negative, p < 0.05" = "#CC6677",
    "Negative, 0.05 <= p < 0.10" = "#E7A1B0",
    "Not significant" = "grey70",
    "Positive, 0.05 <= p < 0.10" = "#88CCAA",
    "Positive, p < 0.05" = "#228833"
  )
  
  # .............................................................................
  # Plot
  # .............................................................................
  if (origin_model == "mixed_model"){
    
    p_volcano <- ggplot(
      volcano_df,
      aes(x = `group:year coeff`, y = neglog10p, color = status)
    ) +
      geom_hline(
        yintercept = -log10(0.05),
        linetype = "dashed",
        linewidth = 0.5,
        color = "grey45"
      ) +
      geom_vline(
        xintercept = 0,
        linetype = "dashed",
        linewidth = 0.5,
        color = "grey55"
      ) +
      geom_point(size = 2.8, alpha = 0.9) +
      geom_text_repel(
        data = label_df,
        aes(label = biomarker),
        size = 3.4,
        box.padding = 0.3,
        point.padding = 0.2,
        segment.color = "grey60",
        max.overlaps = Inf,
        show.legend = FALSE
      ) +
      scale_color_manual(values = volcano_colors) +
      labs(
        title = paste0(test_group_val," effect on biomarker change in ",apoe_name),
        x = paste0("Beta coefficient for ",test_group_val," group"),
        y = expression(-log[10] * "(raw p-value)"),
        color = NULL
      ) +
      theme_classic(base_size = 12) +
      theme(
        plot.title = element_text(face = "bold", size = 12),
        plot.subtitle = element_text(size = 10),
        legend.position = "right",
        legend.text = element_text(size = 9)
      )
  }else{
    
    p_volcano <- ggplot(
      volcano_df,
      aes(x = `group coeff`, y = neglog10p, color = status)
    ) +
      geom_hline(
        yintercept = -log10(0.05),
        linetype = "dashed",
        linewidth = 0.5,
        color = "grey45"
      ) +
      geom_vline(
        xintercept = 0,
        linetype = "dashed",
        linewidth = 0.5,
        color = "grey55"
      ) +
      geom_point(size = 2.8, alpha = 0.9) +
      geom_text_repel(
        data = label_df,
        aes(label = biomarker),
        size = 3.4,
        box.padding = 0.3,
        point.padding = 0.2,
        segment.color = "grey60",
        max.overlaps = Inf,
        show.legend = FALSE
      ) +
      scale_color_manual(values = volcano_colors) +
      labs(
        title = paste0(test_group_val," effect on biomarker change in ",apoe_name),
        x = paste0("Beta coefficient for ",test_group_val," group"),
        y = expression(-log[10] * "(raw p-value)"),
        color = NULL
      ) +
      theme_classic(base_size = 12) +
      theme(
        plot.title = element_text(face = "bold", size = 12),
        plot.subtitle = element_text(size = 10),
        legend.position = "right",
        legend.text = element_text(size = 9)
      )
  }
  
  print(p_volcano)
  
  # .............................................................................
  # Save outputs
  # .............................................................................
  ggsave(
    filename = paste0(output_path,"Stratified/volcano_apoe_",apoe_grp,"_from_",origin_model,"_intervention_effect_biomarker_change_imputed_BMI_adj.png"),
    plot = p_volcano,
    width = 9,
    height = 7,
    dpi = 600
  )
  
  ggsave(
    filename = paste0(output_path,"Stratified/volcano_apoe_",apoe_grp,"_from_",origin_model,"_intervention_effect_biomarker_change_imputed_BMI_adj.svg"),
    plot = p_volcano,
    device = svglite::svglite,
    width = 9,
    height = 7
  )
  
  write_csv(
    volcano_df,
    paste0(output_path,"Stratified/volcano_apoe_",apoe_grp,"_from_",origin_model,"_intervention_effect_biomarker_change_table_imputed.csv")
  )

}

# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
# APOE-stratified ztotal_composite ~ year * baseline biomarker * group ----------------------------------------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////


# APOE-stratified mixed-effects models
#
# Purpose:
# Run the same 3-way interaction model separately within
# APOE4 carrier-status groups.
#
# Outcome:
#   ztotal_composite over time
#
# Main term of interest:
#   year × baseline biomarker × Intervention group group
#
# Stratification variable:
#   apoe4_status
#
# Model within each APOE stratum:
#   ztotal_composite ~ year * baseline biomarker * group +
#     age + sex + education + BMI + site +
#     (1 + year | ID)
#
# Specifications:
# - random intercept + random slope for year: (1 + year | ID)
# - control reference = intervention control
# - sex reference = man 1
# - campid reference = Helsinki 101
# - winsorization at 1% and 99%
# - log transformation after winsorization
# - z-scaling after log transformation
# - require baseline biomarker
# - require baseline cognition and at least one follow-up cognition
# - require complete cofactors



# .............................................................................
# File path
# .............................................................................

file_path <- paste0(processed_path,"dataset_original.csv")
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
  var_group,
  "apoe4_status",
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
# One-biomarker model runner within one APOE stratum
# .............................................................................
run_one_biomarker_stratum <- function(bm_base, data, apoe_level) {
  
  bm0 <- paste0(bm_base, "_0")

  
  #cat("Running:", bm_base, "| APOE stratum:", apoe_level, "\n")
  
  df_model <- data %>%
    filter(apoe4_status == apoe_level) %>%
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
      !is.na(.data[[var_group]]),
      !is.na(apoe4_status)
    ) %>%
    mutate(
      bm0_w = winsorize(.data[[bm0]], p = 0.01)
    )
  
  if (n_distinct(df_model$ID) < 20) {
    return(tibble(
      apoe4_status = as.character(apoe_level),
      biomarker = str_remove(bm_base, "^nulisa_"),
      n_ids = n_distinct(df_model$ID),
      n_obs = NA_integer_,
      error = "Too few participants in APOE stratum"
    ))
  }
  
  if ((needs_log(bm_base) && any(df_model$bm0_w <= 0, na.rm = TRUE))) {
    return(tibble(
      apoe4_status = as.character(apoe_level),
      biomarker = str_remove(bm_base, "^nulisa_"),
      n_ids = n_distinct(df_model$ID),
      n_obs = NA_integer_,
      error = "Non-positive baseline values after winsorization; log not possible"
    ))
  }
  
  df_model <- df_model %>%
    mutate(
      bm0_log = transform_bm(bm0_w, bm_base),
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
    )
  df_long <- df_long %>%
    mutate(
      year = case_when(
        timepoint == "ztotal_composite_BL2"  ~ 0,
        timepoint == "ztotal_composite_12kk" ~ 1,
        timepoint == "ztotal_composite_24kk" ~ 2
      ),
      !!rlang::sym(var_group)  := factor(df_long[[var_group]]),
      sex = factor(sex),
      campid = factor(campid),
      bmi_bl2_z = as.numeric(scale(bmi_bl2))
    ) %>%
    filter(!is.na(ztotal_composite))
  
  if ("3" %in% levels(df_long$sub_sample_id)) {
    df_long$sub_sample_id <- relevel(df_long$sub_sample_id, ref = "3")
  }
    
  if ("1" %in% levels(df_long$sex)) {
    df_long$sex <- relevel(df_long$sex, ref = "1")
  }
  if ("101" %in% levels(df_long$campid)) {
    df_long$campid <- relevel(df_long$campid, ref = "101")
  }
  
  model <- tryCatch(
    lmer(
      paste0(
        "ztotal_composite ~ year * bm0_log_z * ", var_group,
        " + age_bl1 + sex + educ_years_BL1 + bmi_bl2_z + campid +",
        " (1 + year | ID)"
      ),
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
      apoe4_status = as.character(apoe_level),
      biomarker = str_remove(bm_base, "^nulisa_"),
      n_ids = n_distinct(df_long$ID),
      n_obs = nrow(df_long),
      error = model$message
    ))
  }
  
  fixed <- tidy(model, effects = "fixed") %>%
    select(term, estimate, std.error, p.value)
  
  tibble(
    apoe4_status = as.character(apoe_level),
    biomarker = str_remove(bm_base, "^nulisa_"),
    n_ids = n_distinct(df_long$ID),
    n_obs = nrow(df_long),
    
    `bm coeff` = get_term_value(fixed, "bm0_log_z", estimate),
    `bm std` = get_term_value(fixed, "bm0_log_z", std.error),
    `bm pvalue` = get_term_value(fixed, "bm0_log_z", p.value),
    
    `year coeff` = get_term_value(fixed, "year", estimate),
    `year std` = get_term_value(fixed, "year", std.error),
    `year pvalue` = get_term_value(fixed, "year", p.value),
    
    `group coeff` = get_term_value(fixed, ref_group, estimate),
    `group std` = get_term_value(fixed, ref_group, std.error),
    `group pvalue` = get_term_value(fixed, ref_group, p.value),
    
    `bm:year coeff` = get_term_value(fixed, "year:bm0_log_z", estimate),
    `bm:year std` = get_term_value(fixed, "year:bm0_log_z", std.error),
    `bm:year pvalue` = get_term_value(fixed, "year:bm0_log_z", p.value),
    
    `group:year coeff` = get_term_value(fixed, paste0("year:",ref_group), estimate),
    `group:year std` = get_term_value(fixed,   paste0("year:",ref_group), std.error),
    `group:year pvalue` = get_term_value(fixed,paste0("year:",ref_group), p.value),
    
    `group:bm coeff` = get_term_value(fixed, paste0("bm0_log_z:",ref_group), estimate),
    `group:bm std` = get_term_value(fixed, paste0("bm0_log_z:",ref_group), std.error),
    `group:bm pvalue` = get_term_value(fixed, paste0("bm0_log_z:",ref_group), p.value),
    
    `group:bm:year coeff` =
      get_term_value(fixed, paste0("year:bm0_log_z:",ref_group), estimate),
    `group:bm:year std` =
      get_term_value(fixed, paste0("year:bm0_log_z:",ref_group), std.error),
    `group:bm:year pvalue` =
      get_term_value(fixed, paste0("year:bm0_log_z:",ref_group), p.value),
    
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
    
    error = NA_character_
  )
}

# .............................................................................
# Run all biomarkers in each APOE stratum
# .............................................................................
apoe_levels <- sort(unique(df_all$apoe4_status[!is.na(df_all$apoe4_status)]))

results_all <- map_dfr(
  apoe_levels,
  function(apoe_level) {
    map_dfr(
      biomarker_names,
      run_one_biomarker_stratum,
      data = df_all,
      apoe_level = apoe_level, 
      .progress = interactive()
    )
  }
)

# .............................................................................
# Add FDR within each APOE stratum
# .............................................................................
p_cols <- grep("pvalue$", names(results_all), value = TRUE)

results_all <- results_all %>%
  group_by(apoe4_status) %>%
  mutate(
    across(
      all_of(p_cols),
      ~ p.adjust(.x, method = "fdr"),
      .names = "{.col} FDR"
    )
  ) %>%
  ungroup()

# .............................................................................
# Round numeric columns
# .............................................................................
results_all <- results_all %>%
  mutate(across(where(is.numeric), ~ round(.x, 6)))

# .............................................................................
# Save results
# .............................................................................
write_csv(
  results_all,
  paste0(output_path,"Stratified/apoe_stratified_3way_mixed_models_all_biomarkers_log_z_bmi_adj.csv")
)

# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
# APOE-stratified coefficient plot
## 3-way interaction: group × biomarker × year --------------------------------
# Different colors from total-cohort plots
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////


# .............................................................................
# File path
# .............................................................................
file_path <- paste0(output_path, "Stratified/apoe_stratified_3way_mixed_models_all_biomarkers_log_z_bmi_adj.csv")

# .............................................................................
# Proteins to exclude
# .............................................................................
exclude_proteins <- c(
  "pTau181",
  "pTau231",
  "pTau217",
  "AB38",
  "NEFL",
  "GFAP",
  "AB40",
  "lumipulse_ptau217"
)

# .............................................................................
# Read data
# .............................................................................
res <- read_csv(file_path, show_col_types = FALSE)

# .............................................................................
# Prepare plotting data
# .............................................................................
plot_df <- res %>%
  mutate(
    biomarker = str_remove(biomarker, "^nulisa_")
  ) %>%
  filter(!biomarker %in% exclude_proteins) %>%
  transmute(
    apoe4_status,
    biomarker,
    beta = `group:bm:year coeff`,
    se = `group:bm:year std`,
    p = `group:bm:year pvalue`,
    p_fdr = `group:bm:year pvalue FDR`
  ) %>%
  filter(!is.na(beta), !is.na(se), !is.na(p), p < 0.1) %>%
  mutate(
    ci_low = beta - 1.96 * se,
    ci_high = beta + 1.96 * se,
    star = if_else(!is.na(p_fdr) & p_fdr < 0.05, "*", ""),
    sig = case_when(
      p < 0.05 ~ "p < 0.05",
      p < 0.1 ~ "0.05 <= p < 0.1",
      TRUE ~ "NS"
    )
  )

# .............................................................................
# Keep top 20 per APOE stratum
# .............................................................................
plot_df <- plot_df %>%
  group_by(apoe4_status) %>%
  arrange(p, desc(abs(beta)), .by_group = TRUE) %>%
  slice_head(n = 20) %>%
  ungroup()

# .............................................................................
# Symmetric x-axis
# .............................................................................
max_val <- max(abs(c(plot_df$ci_low, plot_df$ci_high)), na.rm = TRUE)
max_val <- max_val * 1.15
x_limits <- c(-max_val, max_val)

# .............................................................................
# .............................................................................
# Plot function with p-value color categories
# .............................................................................
make_apoe_plot <- function(df, apoe_level, dark_color, light_color, panel_title) {
  
  df_sub <- df %>%
    filter(apoe4_status == apoe_level) %>%
    arrange(beta) %>%
    mutate(
      biomarker = factor(biomarker, levels = biomarker),
      point_color = case_when(
        p < 0.05 ~ dark_color,
        p < 0.10 ~ light_color,
        TRUE ~ "grey70"
      )
    )
  
  ggplot(df_sub, aes(x = beta, y = biomarker)) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.5,
      color = "grey45"
    ) +
    geom_errorbarh(
      aes(xmin = ci_low, xmax = ci_high),
      height = 0.18,
      linewidth = 0.6,
      color = "grey45"
    ) +
    geom_point(
      aes(color = point_color),
      size = 3
    ) +
    scale_color_identity() +
    geom_text(
      aes(
        x = ci_high + 0.03 * max_val,
        label = star
      ),
      size = 5,
      fontface = "bold",
      hjust = 0
    ) +
    coord_cartesian(xlim = x_limits) +
    labs(
      title = panel_title,
      subtitle = "Group × biomarker × time",
      caption = "Dark points: raw p < 0.05; light points: 0.05 <= raw p < 0.10; * FDR < 0.05",
      x = expression(beta ~ "coefficient"),
      y = NULL
    ) +
    theme_classic(base_size = 12) +
    theme(
      legend.position = "none",
      plot.title = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(size = 10),
      axis.text.y = element_text(size = 11),
      axis.text.x = element_text(size = 11),
      axis.title.x = element_text(size = 11)
    )
}

# .............................................................................
# Create plots
# .............................................................................
apoe_levels <- sort(unique(plot_df$apoe4_status))

p_noncarrier <- make_apoe_plot(
  plot_df,
  apoe_level = apoe_levels[1],
  dark_color = "#A67708",
  light_color = "#EDB526",
  panel_title = "APOE4 non-carriers"
)

p_carrier <- make_apoe_plot(
  plot_df,
  apoe_level = apoe_levels[2],
  dark_color = "#044D7A",
  light_color = "#38ACEB",
  panel_title = "APOE4 carriers"
)

combined_plot <- p_noncarrier + p_carrier +
  plot_layout(ncol = 2)

print(combined_plot)



# .............................................................................
# Save
# .............................................................................
ggsave(
  filename = paste0(output_path,"Stratified/apoe_stratified_3way_forestplot_different_colours.png"),
  plot = combined_plot,
  width = 13,
  height = 8,
  dpi = 600
)

ggsave(
  filename = paste0(output_path,"Stratified/apoe_stratified_3way_forestplot_different_colours.svg"),
  plot = combined_plot,
  device = svglite::svglite,
  width = 13,
  height = 8
)
write_csv(
  plot_df,
  paste0(output_path,"Stratified/apoe_stratified_3way_forestplot_table.csv")
)







# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
## 4-way interaction with APOE4 status -----------------------------------------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////

# 4-way interaction with APOE4 status
#Mixed-effects models for all baseline NULISA biomarkers
# Output: one row per biomarker, columns = beta / SE / raw p / FDR
#
# Specifications:
# - random intercept + random slope for year: (1 + year | ID)
# - campid reference = Helsinki (101)
# - group reference = control (0)
# - sex reference = man (1)
# - winsorization at 1% and 99%, baseline and 2-year separately
# - log transformation AFTER winsorization
# - z-scaling AFTER log transformation
# - remove patients without baseline biomarker
# - remove patients without baseline cognition OR without at least one follow-up
# - remove patients without one of the cofactors


# .............................................................................
# File path
# .............................................................................
file_path <- paste0(processed_path,"dataset_original.csv")

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
  var_group,
  "age_bl1",
  "sex",
  "educ_years_BL1",
  "apoe4_status",
  "bmi_bl2",
  "campid",
  "ztotal_composite_BL2",
  "ztotal_composite_12kk",
  "ztotal_composite_24kk"
)



# .............................................................................
# One-biomarker model runner
# .............................................................................
run_one_biomarker <- function(bm_base, data) {
  
  bm0 <- paste0(bm_base, "_0")
  bm2 <- paste0(bm_base, "_2")

  
  cat("Running:", bm_base, "\n") 
  
  df_model <- data %>%
    select(all_of(c(base_vars, bm0, bm2))) %>%
    filter(
      !is.na(.data[[bm0]]),
      !is.na(ztotal_composite_BL2),
      (!is.na(ztotal_composite_12kk) | !is.na(ztotal_composite_24kk)),
      !is.na(age_bl1),
      !is.na(sex),
      !is.na(educ_years_BL1),
      !is.na(bmi_bl2),
      !is.na(apoe4_status),
      !is.na(campid),
      !is.na(.data[[var_group]])
    ) %>%
    mutate(
      bm0_w = winsorize(.data[[bm0]], p = 0.01),
      bm2_w = winsorize(.data[[bm2]], p = 0.01)
    )
  
  # Log requires positive values at both timepoints
  if ((needs_log(bm_base) && any(df_model$bm0_w <= 0, na.rm = TRUE)) ||
      (needs_log(bm_base) && any(df_model$bm2_w <= 0, na.rm = TRUE))) {
    return(tibble(
      biomarker = str_remove(bm_base, "^nulisa_"),
      n_ids = n_distinct(df_model$ID),
      n_obs = NA_integer_,
      error = "Non-positive values after winsorization; log not possible"
    ))
  }
  
  df_model <- df_model %>%
    mutate(
      bm0_log   = transform_bm(bm0_w, bm_base),
      bm2_log   = transform_bm(bm2_w, bm_base),
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
    ) 
  df_long <- df_long %>%
    mutate(
      year = case_when(
        timepoint == "ztotal_composite_BL2"  ~ 0,
        timepoint == "ztotal_composite_12kk" ~ 1,
        timepoint == "ztotal_composite_24kk" ~ 2
      ),      
      !!rlang::sym(var_group)  := factor(df_long[[var_group]]),
      apoe4_status = factor(apoe4_status),
      sex = factor(sex),
      campid = factor(campid),
      bmi_bl2_z = as.numeric(scale(bmi_bl2))
    ) %>%
    filter(!is.na(ztotal_composite))
  
  # Reference groups

  if ("3" %in% levels(df_long$sub_sample_id)) {
    df_long$sub_sample_id <- relevel(df_long$sub_sample_id, ref = "3")
  }
  if ("1" %in% levels(df_long$sex)) {
    df_long$sex <- relevel(df_long$sex, ref = "1")
  }
  if ("101" %in% levels(df_long$campid)) {
    df_long$campid <- relevel(df_long$campid, ref = "101")
  }
  
  if ("0" %in% levels(df_long$apoe4_status)) {
    df_long$apoe4_status <- relevel(df_long$apoe4_status, ref = "0")
  }
  
  model <- tryCatch(
    lmer(
        paste0(
        "ztotal_composite ~ year * bm0_log_z * ", var_group,
        "* apoe4_status + age_bl1 + sex + educ_years_BL1 + bmi_bl2_z + campid +",
        " (1 + year | ID)"
      ),
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
  
  
  control_noncarrier_slope <- get_linear_combo(
    model,
    terms = c("year:bm0_log_z")
  )
  
  intervention_noncarrier_slope <- get_linear_combo(
    model,
    terms = c(
      "year:bm0_log_z",
      paste0("year:bm0_log_z:",ref_group)
    )
  )
  
  control_carrier_slope <- get_linear_combo(
    model,
    terms = c(
      "year:bm0_log_z",
      "year:bm0_log_z:apoe4_status1"
    )
  )
  
  intervention_carrier_slope <- get_linear_combo(
    model,
    terms = c(
      "year:bm0_log_z",
      paste0( "year:bm0_log_z:",ref_group),
      "year:bm0_log_z:apoe4_status1",
      paste0( "year:bm0_log_z:",ref_group,":apoe4_status1")
    )
  )
  
  tibble(
    biomarker = str_remove(bm_base, "^nulisa_"),
    n_ids = n_distinct(df_long$ID),
    n_obs = nrow(df_long),
    
    `Intercept coeff` = get_term_value(fixed, "(Intercept)", estimate),
    `Intercept std`   = get_term_value(fixed, "(Intercept)", std.error),
    `Intercept pvalue`= get_term_value(fixed, "(Intercept)", p.value),
    
    `group coeff` = get_term_value(fixed, ref_group, estimate),
    `group std`   = get_term_value(fixed, ref_group, std.error),
    `group pvalue`= get_term_value(fixed, ref_group, p.value),
    
    `bm coeff` = get_term_value(fixed, "bm0_log_z", estimate),
    `bm std`   = get_term_value(fixed, "bm0_log_z", std.error),
    `bm pvalue`= get_term_value(fixed, "bm0_log_z", p.value),
    
    `group:bm coeff` = get_term_value(fixed, paste0("bm0_log_z:",ref_group), estimate),
    `group:bm std`   = get_term_value(fixed, paste0("bm0_log_z:",ref_group), std.error),
    `group:bm pvalue`= get_term_value(fixed, paste0("bm0_log_z:",ref_group), p.value),
    
    `year coeff` = get_term_value(fixed, "year", estimate),
    `year std`   = get_term_value(fixed, "year", std.error),
    `year pvalue`= get_term_value(fixed, "year", p.value),
    
    `group:year coeff` = get_term_value(fixed, paste0("year:",ref_group), estimate),
    `group:year std`   = get_term_value(fixed, paste0("year:",ref_group), std.error),
    `group:year pvalue`= get_term_value(fixed, paste0("year:",ref_group), p.value),
    
    `bm:year coeff` = get_term_value(fixed, "year:bm0_log_z", estimate),
    `bm:year std`   = get_term_value(fixed, "year:bm0_log_z", std.error),
    `bm:year pvalue`= get_term_value(fixed, "year:bm0_log_z", p.value),
    
    `group:bm:year coeff` =
      get_term_value(fixed, paste0("year:bm0_log_z:",ref_group), estimate),
    `group:bm:year std` =
      get_term_value(fixed, paste0("year:bm0_log_z:",ref_group), std.error),
    `group:bm:year pvalue` =
      get_term_value(fixed, paste0("year:bm0_log_z:",ref_group), p.value),
    
    `group:bm:year:apoe4 coeff` =
      get_term_value(
        fixed,
        paste0("year:bm0_log_z:",ref_group,":apoe4_status1"),
        estimate
      ),
    
    `group:bm:year:apoe4 std` =
      get_term_value(
        fixed,
        paste0("year:bm0_log_z:",ref_group,":apoe4_status1"),
        std.error
      ),
    
    `group:bm:year:apoe4 pvalue` =
      get_term_value(
        fixed,
        paste0("year:bm0_log_z:",ref_group,":apoe4_status1"),
        p.value
      ),
    `control noncarrier bm:year coeff` = control_noncarrier_slope$estimate,
    `control noncarrier bm:year std` = control_noncarrier_slope$std.error,
    `control noncarrier bm:year pvalue` = control_noncarrier_slope$p.value,
    `control noncarrier bm:year ci_low` = control_noncarrier_slope$ci_low,
    `control noncarrier bm:year ci_high` = control_noncarrier_slope$ci_high,
    
    !!glue::glue("{interv_val_group} noncarrier bm:year coeff") := intervention_noncarrier_slope$estimate,
    !!glue::glue("{interv_val_group} noncarrier bm:year std") := intervention_noncarrier_slope$std.error,
    !!glue::glue("{interv_val_group} noncarrier bm:year pvalue") := intervention_noncarrier_slope$p.value,
    !!glue::glue("{interv_val_group} noncarrier bm:year ci_low" ):= intervention_noncarrier_slope$ci_low,
    !!glue::glue("{interv_val_group} noncarrier bm:year ci_high") := intervention_noncarrier_slope$ci_high,
    
    `control carrier bm:year coeff` = control_carrier_slope$estimate,
    `control carrier bm:year std` = control_carrier_slope$std.error,
    `control carrier bm:year pvalue` = control_carrier_slope$p.value,
    `control carrier bm:year ci_low` = control_carrier_slope$ci_low,
    `control carrier bm:year ci_high` = control_carrier_slope$ci_high,
    
    !!glue::glue("{interv_val_group} carrier bm:year coeff") := intervention_carrier_slope$estimate,
    !!glue::glue("{interv_val_group} carrier bm:year std") := intervention_carrier_slope$std.error,
    !!glue::glue("{interv_val_group} carrier bm:year pvalue") := intervention_carrier_slope$p.value,
    !!glue::glue("{interv_val_group} carrier bm:year ci_low") := intervention_carrier_slope$ci_low,
    !!glue::glue("{interv_val_group} carrier bm:year ci_high") := intervention_carrier_slope$ci_high,
    `age_0y coeff` = get_term_value(fixed, "age_bl1", estimate),
    `age_0y std`   = get_term_value(fixed, "age_bl1", std.error),
    `age_0y pvalue`= get_term_value(fixed, "age_bl1", p.value),
    
    `bmi_0y coeff` = get_term_value(fixed, "bmi_bl2_z", estimate),
    `bmi_0y std`   = get_term_value(fixed, "bmi_bl2_z", std.error),
    `bmi_0y pvalue`= get_term_value(fixed, "bmi_bl2_z", p.value),
    
    
    `sex coeff` = get_term_value(fixed, "sex2", estimate),
    `sex std`   = get_term_value(fixed, "sex2", std.error),
    `sex pvalue`= get_term_value(fixed, "sex2", p.value),
    
    `educ_years_0y coeff` =
      get_term_value(fixed, "educ_years_BL1", estimate),
    `educ_years_0y std` =
      get_term_value(fixed, "educ_years_BL1", std.error),
    `educ_years_0y pvalue` =
      get_term_value(fixed, "educ_years_BL1", p.value),
    
    error = NA_character_
  )
}
# .............................................................................
# Run all models
# .............................................................................
results_all <- map_dfr(biomarker_names, run_one_biomarker, data = df_all, .progress = interactive())

# .............................................................................
# Add FDR columns
# .............................................................................
p_cols <- grep("pvalue$", names(results_all), value = TRUE)

for (pcol in p_cols) {
  fdr_col <- paste0(pcol, " FDR")
  results_all[[fdr_col]] <- p.adjust(results_all[[pcol]], method = "fdr")
}

# .............................................................................
# Round numeric columns
# .............................................................................
results_all <- results_all %>%
  mutate(across(where(is.numeric), ~ round(.x, 6)))

# .............................................................................
# Save results
# .............................................................................
write_csv(
  results_all,
  paste0(output_path,"Stratified/ztotal_composite_change_4way_mixed_model_baseline_bm_log_z_imputed_zeros_bmi_adj.csv")
)




# .............................................................................
### Top bm ----
# .............................................................................

results_all = read_csv(paste0(output_path,"Stratified/ztotal_composite_change_4way_mixed_model_baseline_bm_log_z_imputed_zeros_bmi_adj.csv"))

top25_tables <- create_top25_tables(
  results = results_all,
  output_path = output_path,
  folder = "Stratified",
  method = "group_bm_year_apoe",
  p = "group:bm:year:apoe4 pvalue"
)

top25_df <- top25_tables$top25_df
top25_df_low <- top25_tables$top25_df_low




# /////////////////////////////////////////////////////////
### Volcano plot  ----------------
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
  folder = "Stratified",
  output_file_name = "volcano_ztotal_vs_time_vs_bbm_vs_group_vs_apoe",
  title = "Volcano plot of 4-way interaction",
  results_filtered = results_filtered,
  top25_df = top25_df,
  p = "group:bm:year:apoe4 pvalue",
  beta = "group:bm:year:apoe4 coeff"
  
)



# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
## Coefficient plots for mixed model results ----------------------------------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////

# Coefficient plots for mixed model results
#
# Outputs:
#   1) Old two-panel plot:
#      A) baseline biomarker × time
#      B) group × baseline biomarker × time
#
#   2) New combined plot:
#      For each biomarker, two estimates are shown:
#      - Control: baseline biomarker × time
#      - Intervention group: control slope + group × biomarker × time
#
# New combined plot uses accurate SE/CI from model-derived
# linear combinations:
#   control bm:year coeff / ci_low / ci_high
#   intervention bm:year coeff / ci_low / ci_high



# .............................................................................
# File path
# .............................................................................
file_path <- paste0(output_path,"Stratified/ztotal_composite_change_4way_mixed_model_baseline_bm_log_z_imputed_zeros_bmi_adj.csv")

# .............................................................................
# Proteins to exclude
# .............................................................................
exclude_proteins <- c(
  "pTau181",
  "pTau231",
  "pTau217",
  "AB38",
  "NEFL",
  "GFAP",
  "AB40",
  "lumipulse_ptau217"
)

# .............................................................................
# Read data
# .............................................................................
res <- read_csv(file_path, show_col_types = FALSE)

if (!"Protein" %in% names(res)) {
  names(res)[1] <- "Protein"
}

res <- res %>%
  mutate(
    Protein = str_remove(Protein, "^nulisa_")
  ) %>%
  filter(!Protein %in% exclude_proteins)

# ..............................................................................
# OLD FIGURE: separate coefficient plots
# .............................................................................

build_term_table <- function(data, beta_col, se_col, p_col,
                             top_n = 25, p_cutoff = 0.1) {
  data %>%
    transmute(
      Protein,
      beta = .data[[beta_col]],
      se   = .data[[se_col]],
      p    = .data[[p_col]]
    ) %>%
    filter(!is.na(beta), !is.na(se), !is.na(p)) %>%
    mutate(
      ci_low  = beta - 1.96 * se,
      ci_high = beta + 1.96 * se,
      sig = case_when(
        p < 0.05 ~ "p < 0.05",
        p < p_cutoff ~ "0.05 <= p < 0.1",
        TRUE ~ "NS"
      ),
      abs_beta = abs(beta)
    ) %>%
    filter(p < p_cutoff) %>%
    arrange(p, desc(abs_beta)) %>%
    slice_head(n = top_n) %>%
    arrange(beta) %>%
    mutate(Protein = factor(Protein, levels = Protein))
}

plot_bmyear <- build_term_table(
  data = res,
  beta_col = "bm:year coeff",
  se_col   = "bm:year std",
  p_col    = "bm:year pvalue",
  top_n    = 25,
  p_cutoff = 0.1
)

plot_3way <- build_term_table(
  data = res,
  beta_col = "group:bm:year coeff",
  se_col   = "group:bm:year std",
  p_col    = "group:bm:year pvalue",
  top_n    = 25,
  p_cutoff = 0.1
)

max_val_old <- max(
  abs(c(
    plot_bmyear$ci_low,
    plot_bmyear$ci_high,
    plot_3way$ci_low,
    plot_3way$ci_high
  )),
  na.rm = TRUE
)

x_limits_old <- c(-1.08 * max_val_old, 1.08 * max_val_old)

make_coef_plot <- function(df, panel_label, title, subtitle) {
  ggplot(df, aes(x = beta, y = Protein, color = sig)) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.5,
      color = "grey40"
    ) +
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
        "0.05 <= p < 0.1" = "#6baed6",
        "NS" = "grey70"
      ),
      drop = FALSE
    ) +
    coord_cartesian(xlim = x_limits_old) +
    labs(
      title = paste0(panel_label, "  ", title),
      subtitle = subtitle,
      x = "Beta coefficient",
      y = NULL,
      color = NULL
    ) +
    theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      plot.subtitle = element_text(size = 10),
      axis.text.y = element_text(size = 11),
      axis.text.x = element_text(size = 11),
      axis.title.x = element_text(size = 11),
      legend.position = "top",
      legend.text = element_text(size = 9)
    )
}

p1 <- make_coef_plot(
  df = plot_bmyear,
  panel_label = "A",
  title = "Baseline biomarker association with cognitive change",
  subtitle = paste0(
    "Top 25 biomarkers ranked by p value",
    "\nInteraction term: baseline biomarker x time"
  )
)

p2 <- make_coef_plot(
  df = plot_3way,
  panel_label = "B",
  title = paste0(test_group_val," modification of biomarker-associated cognitive change"),
  subtitle = paste0(
    "Top 25 biomarkers ranked by p value",
    paste0("\n",test_group_val," term: group x baseline biomarker x time")
  )
)

old_combined_plot <- p1 + p2 + patchwork::plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "top")

print(old_combined_plot)

ggsave(
  filename = paste0(output_path,"Stratified/mixed_models_top25_old_two_panel_forest.png"),
  plot = old_combined_plot,
  width = 13,
  height = 9,
  dpi = 600,
  bg = "white"
)

ggsave(
  filename = paste0(output_path,"Stratified/mixed_models_top25_old_two_panel_forest.svg"),
  plot = old_combined_plot,
  device = svglite::svglite,
  width = 13,
  height = 9
)

ggsave(
  filename = paste0(output_path,"Stratified/mixed_models_top25_old_two_panel_forest.pdf"),
  plot = old_combined_plot,
  width = 13,
  height = 9,
  bg = "white"
)


# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
## 4-WAY FIGURE WITH TWO PANELS — REFINED ----------------------------------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////

# Panel A:
#   Formal 4-way interaction effect:
#   group x biomarker x time x APOE4
#
# Panel B:
#   Exact subgroup-specific slopes:
#   Control / Intervention groupx APOE4 non-carrier / carrier
#
# Main figure includes only biomarkers with raw 4-way p < 0.05.
# FDR-significant 4-way interactions are marked with **.
# Raw p < 0.05 are marked with *.

required_4way_cols <- c(
  "control noncarrier bm:year coeff",
  "control noncarrier bm:year pvalue",
  "control noncarrier bm:year ci_low",
  "control noncarrier bm:year ci_high",
  paste0(interv_val_group, " noncarrier bm:year coeff"),
  paste0(interv_val_group, " noncarrier bm:year pvalue"),
  paste0(interv_val_group, " noncarrier bm:year ci_low"),
  paste0(interv_val_group, " noncarrier bm:year ci_high"),
  "control carrier bm:year coeff",
  "control carrier bm:year pvalue",
  "control carrier bm:year ci_low",
  "control carrier bm:year ci_high",
  paste0(interv_val_group, " carrier bm:year coeff"),
  paste0(interv_val_group, " carrier bm:year pvalue"),
  paste0(interv_val_group, " carrier bm:year ci_low"),
  paste0(interv_val_group, " carrier bm:year ci_high"),
  "group:bm:year:apoe4 coeff",
  "group:bm:year:apoe4 std",
  "group:bm:year:apoe4 pvalue"
)

missing_4way_cols <- setdiff(required_4way_cols, names(res))

if (length(missing_4way_cols) > 0) {
  warning(
    paste0(
      "4-way plot skipped because these columns are missing:\n",
      paste(missing_4way_cols, collapse = "\n"),
      "\n\nRe-run the 4-way mixed model script after adding exact APOE/group-specific linear combinations."
    )
  )
} else {
  
  fourway_fdr_col <- "group:bm:year:apoe4 pvalue FDR"
  has_fourway_fdr <- fourway_fdr_col %in% names(res)
  
  txt_carrier <- paste0("Stronger ",test_group_val, " modification in APOE4 carriers")
  txt_noncarrier <- paste0("Stronger ",test_group_val, " modification in APOE4 non-carriers")
  
  
  if (!has_fourway_fdr) {
    message("No 4-way FDR column found. Only raw p-value significance will be marked.")
  }
  
  plot_4way_wide <- res %>%
    dplyr::transmute(
      Protein,
      fourway_beta = `group:bm:year:apoe4 coeff`,
      fourway_se   = `group:bm:year:apoe4 std`,
      fourway_p    = `group:bm:year:apoe4 pvalue`,
      fourway_fdr  = if (has_fourway_fdr) res[[fourway_fdr_col]] else NA_real_,
      
      control_noncarrier_beta    = `control noncarrier bm:year coeff`,
      control_noncarrier_p       = `control noncarrier bm:year pvalue`,
      control_noncarrier_ci_low  = `control noncarrier bm:year ci_low`,
      control_noncarrier_ci_high = `control noncarrier bm:year ci_high`,
      
      intervention_noncarrier_beta    = res[[paste0(interv_val_group, " noncarrier bm:year coeff")]],
      intervention_noncarrier_p       = res[[paste0(interv_val_group, " noncarrier bm:year pvalue")]],
      intervention_noncarrier_ci_low  = res[[paste0(interv_val_group, " noncarrier bm:year ci_low")]],
      intervention_noncarrier_ci_high = res[[paste0(interv_val_group, " noncarrier bm:year ci_high")]],
      
      control_carrier_beta    = `control carrier bm:year coeff`,
      control_carrier_p       = `control carrier bm:year pvalue`,
      control_carrier_ci_low  = `control carrier bm:year ci_low`,
      control_carrier_ci_high = `control carrier bm:year ci_high`,
      
      intervention_carrier_beta    = res[[paste0(interv_val_group, " carrier bm:year coeff")]],
      intervention_carrier_p       = res[[paste0(interv_val_group, " carrier bm:year pvalue")]],
      intervention_carrier_ci_low  = res[[paste0(interv_val_group, " carrier bm:year ci_low")]],
      intervention_carrier_ci_high = res[[paste0(interv_val_group, " carrier bm:year ci_high")]]
      
    ) %>%
    dplyr::filter(
      !is.na(fourway_beta),
      !is.na(fourway_se),
      !is.na(fourway_p)
    ) %>%
    dplyr::mutate(
      fourway_ci_low  = fourway_beta - 1.96 * fourway_se,
      fourway_ci_high = fourway_beta + 1.96 * fourway_se,
      
      apoe_difference = abs(
        intervention_carrier_beta - control_carrier_beta
      ) - abs(
        intervention_noncarrier_beta - control_noncarrier_beta
      ),
      
      direction_block = dplyr::case_when(
        apoe_difference >= 0 ~ txt_carrier,
        apoe_difference < 0  ~ txt_noncarrier
      ),
      
      fourway_marker = dplyr::case_when(
        !is.na(fourway_fdr) & fourway_fdr < 0.05 ~ "**",
        fourway_p < 0.05 ~ "*",
        TRUE ~ ""
      )
    ) %>%
    dplyr::filter(fourway_p < 0.05) %>%
    dplyr::arrange(dplyr::desc(abs(fourway_beta)))
  
  if (nrow(plot_4way_wide) == 0) {
    stop("No biomarkers with raw 4-way p < 0.05 after filtering.")
  }
  
  protein_order_4way <- plot_4way_wide %>%
    dplyr::arrange(fourway_beta) %>%
    dplyr::pull(Protein)
  
  plot_4way_wide <- plot_4way_wide %>%
    dplyr::mutate(
      Protein = factor(Protein, levels = protein_order_4way)
    )
  
  p_A <- ggplot(plot_4way_wide, aes(x = fourway_beta, y = Protein)) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.55,
      color = "grey45"
    ) +
    geom_errorbarh(
      aes(xmin = fourway_ci_low, xmax = fourway_ci_high),
      height = 0.22,
      linewidth = 0.90,
      color = "grey35"
    ) +
    geom_point(
      aes(color = direction_block),
      size = 4.3,
      shape = 18
    ) +
    geom_text(
      aes(
        x = dplyr::if_else(
          fourway_beta >= 0,
          fourway_ci_high + 0.004,
          fourway_ci_low - 0.004
        ),
        label = fourway_marker
      ),
      size = 4.3,
      fontface = "bold",
      show.legend = FALSE
    ) +
    scale_color_manual(
      values = setNames(
        c(color_carrier, color_noncarrier),
        c(
          paste0("Stronger ", test_group_val, " modification in APOE4 carriers"),
          paste0("Stronger ", test_group_val, " modification in APOE4 non-carriers")
        )
      )
    )+
    
    coord_cartesian(
      xlim = c(-0.14, 0.14),
      clip = "off"
    ) +
    labs(
      title = paste0("A. APOE4-dependent ",test_group_val," modification"),
      x = "4-way beta coefficient",
      y = NULL,
      color = NULL
    ) +
    guides(
      color = guide_legend(nrow = 2, byrow = TRUE)
    ) +
    
    theme_classic(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 10.5, color = "grey35"),
      axis.text.y = element_text(size = 11),
      axis.text.x = element_text(size = 11),
      axis.title.x = element_text(size = 11),
      legend.position = "top",
      legend.justification = "left",
      legend.direction = "vertical",
      legend.margin = margin(0, 0, -3, 0),
      legend.box.spacing = unit(0.2, "lines"),
      legend.title.align = 0,
      legend.text.align = 0,
      legend.text = element_text(size = 8),
      plot.margin = margin(10, 25, 10, 10)
    )
  
  
  plot_4way_long <- plot_4way_wide %>%
    tidyr::pivot_longer(
      cols = c(
        control_noncarrier_beta,
        control_noncarrier_p,
        control_noncarrier_ci_low,
        control_noncarrier_ci_high,
        intervention_noncarrier_beta,
        intervention_noncarrier_p,
        intervention_noncarrier_ci_low,
        intervention_noncarrier_ci_high,
        control_carrier_beta,
        control_carrier_p,
        control_carrier_ci_low,
        control_carrier_ci_high,
        intervention_carrier_beta,
        intervention_carrier_p,
        intervention_carrier_ci_low,
        intervention_carrier_ci_high
      ),
      names_to = c("TrialGroup", "APOE4", ".value"),
      names_pattern = "(control|intervention)_(noncarrier|carrier)_(beta|p|ci_low|ci_high)"
    ) %>%
    dplyr::mutate(
      TrialGroup = dplyr::recode(
        TrialGroup,
        control = "Control",
        intervention = test_group_val
      ),
      APOE4 = dplyr::recode(
        APOE4,
        noncarrier = "APOE4 non-carrier",
        carrier = "APOE4 carrier"
      ),
      TrialGroup = factor(TrialGroup, levels = c("Control",test_group_val)),
      APOE4 = factor(APOE4, levels = c("APOE4 non-carrier", "APOE4 carrier")),
      
      subgroup_marker = dplyr::if_else(p < 0.05, "*", ""),
      label_x = dplyr::case_when(
        beta >= 0 ~ ci_high + 0.0015,
        beta < 0  ~ ci_low  - 0.0015
      )
    )
  plot_4way_long$TrialGroup <- factor(
    plot_4way_long$TrialGroup,
    levels = c("Control", test_group_val)
  )
  
  plot_4way_long$APOE4 <- factor(
    plot_4way_long$APOE4,
    levels = c("APOE4 non-carrier", "APOE4 carrier")
  )
  color_val = c(
    "Control" = color_control,
    "Intervention" = color_interv
  )
  # Create a new variable to define the order of groups
  plot_4way_long$group_order <- factor(
    paste(plot_4way_long$APOE4, plot_4way_long$TrialGroup, sep = "_"),
    levels = c(
      "APOE4 non-carrier_Control",
      paste0("APOE4 non-carrier_",test_group_val),
      "APOE4 carrier_Control",
      paste0("APOE4 carrier_",test_group_val)
    )
  )
  
  p_B <- ggplot(
    plot_4way_long,
    aes(
      x = beta,
      y = Protein,
      color = TrialGroup,
      shape = APOE4,
      group = group_order  # Ensure grouping follows the desired order
    )
  ) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.55,
      color = "grey45"
    ) +
    geom_errorbarh(
      aes(xmin = ci_low, xmax = ci_high, linetype = APOE4),
      height = 0.22,
      linewidth = 0.90,
      position = position_dodge(width = 0.75)
    ) +
    geom_point(
      size = 2.7,
      position = position_dodge(width = 0.75)
    ) +
    geom_text(
      aes(x = label_x, label = subgroup_marker),
      position = position_dodge(width = 0.75),
      size = 4.3,
      fontface = "bold",
      show.legend = FALSE
    ) +
    scale_color_manual(
      values = color_val
    ) +
    scale_shape_manual(
      values = c(
        "APOE4 non-carrier" = 16,
        "APOE4 carrier" = 17
      )
    ) +
    scale_linetype_manual(
      values = c(
        "APOE4 non-carrier" = "dashed",  # Dotted lines for non-carriers
        "APOE4 carrier" = "solid"         # Solid lines for carriers
      )
    ) +
    coord_cartesian(
      xlim = c(-0.08, 0.08),
      clip = "off"
    ) +
    labs(
      title = "B. APOE- and group-specific biomarker slopes",
      x = "Estimated beta coefficient for cognitive change",
      y = NULL,
      color = "Trial group",
      shape = "APOE4 status",
      linetype = "APOE4 status"  # Add legend for linetype
    ) +
    theme_classic(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 10.5, color = "grey35"),
      axis.text.y = element_text( size = 11),
      axis.ticks.y = element_blank(),
      axis.text.x = element_text(size = 11),
      axis.title.x = element_text(size = 11),
      legend.position = "top",
      legend.justification = "left",
      legend.margin = margin(0, 0, -3, 0),
      legend.box.spacing = unit(0.2, "lines"),
      legend.title.align = 0,
      legend.text.align = 0,
      legend.text = element_text(size = 8),
      legend.title = element_text(size = 9),
      plot.margin = margin(10, 25, 10, 5),
      #legend.box = "vertical",  # Stack legends vertically
      #legend.spacing.x = unit(0,"cm"),
      legend.key.size = unit(1.5, "cm")  # Increase the size of the legend key area
      
    ) 
  
    combined_4way <- p_A + p_B +
    patchwork::plot_layout(
      ncol = 2,
      widths = c(0.95, 1.35)
    ) &
    theme(
      panel.spacing = unit(1.5, "lines")
    )
  
  print(combined_4way)
  
  fig_height_4way <- max(9, 0.55 * nrow(plot_4way_wide))
  
  ggsave(
    filename = paste0(output_path,"Stratified/mixed_models_4way_APOE_two_panel_interaction_and_slopes_p005_refined.png"),
    plot = combined_4way,
    width = 15,
    height = fig_height_4way,
    dpi = 600,
    bg = "white"
  )
  
  ggsave(
    filename = paste0(output_path,"Stratified/mixed_models_4way_APOE_two_panel_interaction_and_slopes_p005_refined.svg"),
    plot = combined_4way,
    device = svglite::svglite,
    width = 15,
    height = fig_height_4way
  )
  
  ggsave(
    filename = paste0(output_path,"Stratified/mixed_models_4way_APOE_two_panel_interaction_and_slopes_p005_refined.pdf"),
    plot = combined_4way,
    width = 15,
    height = fig_height_4way,
    bg = "white"
  )
  
  readr::write_csv(
    plot_4way_wide,
    paste0(output_path,"Stratified/mixed_models_4way_APOE_panelA_formal_interaction_p005.csv")
  )
  
  readr::write_csv(
    plot_4way_long,
    paste0(output_path,"Stratified/mixed_models_4way_APOE_panelB_subgroup_slopes_p005.csv")
  )
}

# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
## SUPPLEMENTARY TABLE FOR 4-WAY APOE INTERACTION FIGURE ----------------------------------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
#
# Includes:
# - Formal 4-way interaction statistics
# - FDR correction
# - Exact subgroup-specific slopes
# - Direction labels
# -  Includes biomarkers with nominal 4-way p < 0.10
# Output:
#   Supplementary_Table_APOE4_4way_interactions_p010.csv


# .............................................................................
# File path
# .............................................................................
file_path <- paste0(output_path,"Stratified/ztotal_composite_change_4way_mixed_model_baseline_bm_log_z_imputed_zeros_bmi_adj.csv")

# .............................................................................
# Read results
# .............................................................................
res <- read_csv(file_path, show_col_types = FALSE)

if (!"Protein" %in% names(res)) {
  names(res)[1] <- "Protein"
}

res <- res %>%
  mutate(
    Protein = str_remove(Protein, "^nulisa_")
  )

# .............................................................................
# Detect optional FDR column
# .............................................................................
fourway_fdr_col <- "group:bm:year:apoe4 pvalue FDR"
has_fdr <- fourway_fdr_col %in% names(res)

# .............................................................................
# Build supplementary table
# .............................................................................
supp_table <- res %>%
  transmute(
    Protein,
    
    fourway_beta = `group:bm:year:apoe4 coeff`,
    fourway_se   = `group:bm:year:apoe4 std`,
    fourway_p    = `group:bm:year:apoe4 pvalue`,
    fourway_fdr  = if (has_fdr) res[[fourway_fdr_col]] else NA_real_,
    
    fourway_ci_low  = fourway_beta - 1.96 * fourway_se,
    fourway_ci_high = fourway_beta + 1.96 * fourway_se,
    abs_fourway_beta = abs(fourway_beta),
    
    control_noncarrier_beta = `control noncarrier bm:year coeff`,
    control_noncarrier_p    = `control noncarrier bm:year pvalue`,

    intervention_noncarrier_beta = res[[paste0(interv_val_group, " noncarrier bm:year coeff")]],
    intervention_noncarrier_p    = res[[paste0(interv_val_group, " noncarrier bm:year pvalue")]],
    
    control_carrier_beta = `control carrier bm:year coeff`,
    control_carrier_p    = `control carrier bm:year pvalue`,
    
    intervention_carrier_beta = res[[paste0(interv_val_group, " carrier bm:year coeff")]],
    intervention_carrier_p    = res[[paste0(interv_val_group, " carrier bm:year pvalue")]]
  ) %>%
  mutate(
    apoe_difference = abs(
      intervention_carrier_beta - control_carrier_beta
    ) - abs(
      intervention_noncarrier_beta - control_noncarrier_beta
    ),
    
    interaction_direction = case_when(
      apoe_difference > 0 ~
        paste0("Stronger ",test_group_val,"-related biomarker association in APOE4 carriers"),
      apoe_difference < 0 ~
        paste0("Stronger ",test_group_val,"-related biomarker association in APOE4 non-carriers"),
      TRUE ~
        "No directional difference"
    ),
    
    significance_label = case_when(
      !is.na(fourway_fdr) & fourway_fdr < 0.05 ~ "FDR significant",
      fourway_p < 0.05 ~ "Nominal p < 0.05",
      fourway_p < 0.10 ~ "Nominal p 0.05-0.10",
      TRUE ~ "Not significant"
    )
  ) %>%
  filter(fourway_p < 0.10) %>%
  arrange(fourway_p)

# .............................................................................
# Round numeric columns
# .............................................................................
supp_table <- supp_table %>%
  mutate(
    across(where(is.numeric), ~ round(.x, 4))
  )

# .............................................................................
# Save table
# .............................................................................
write_csv(
  supp_table,
  paste0(output_path,"Stratified/Supplementary_Table_APOE4_4way_interactions_p010.csv")
)



# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
# Sex-stratified ztotal_composite ~ year * baseline biomarker * group ----------------------------------------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////


# SEX-stratified mixed-effects models
#
# Purpose:
# Run the same 3-way interaction model separately within
# Sex-status groups.
#
# Outcome:
#   ztotal_composite over time
#
# Main term of interest:
#   year × baseline biomarker × Intervention group group
#
# Stratification variable:
#   sex
#
# Model within each SEX stratum:
#   ztotal_composite ~ year * baseline biomarker * group +
#     age + apoe + education + BMI + site +
#     (1 + year | ID)
#
# Specifications:
# - random intercept + random slope for year: (1 + year | ID)
# - control reference =  control
# - apoe reference = non carrier
# - campid reference = Helsinki 101
# - winsorization at 1% and 99%
# - log transformation after winsorization
# - z-scaling 
# - require baseline biomarker
# - require baseline cognition and at least one follow-up cognition
# - require complete cofactors



# .............................................................................
# File path
# .............................................................................

file_path <- paste0(processed_path,"dataset_original.csv")
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
  var_group,
  "apoe4_status",
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
# One-biomarker model runner within one APOE stratum
# .............................................................................
run_one_biomarker_stratum <- function(bm_base, data, sex_level) {
  
  bm0 <- paste0(bm_base, "_0")
  
  df_model <- data %>%
    filter(sex == sex_level) %>%
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
      !is.na(.data[[var_group]]),
      !is.na(apoe4_status)
    ) %>%
    mutate(
      bm0_w = winsorize(.data[[bm0]], p = 0.01)
    )
  
  if (n_distinct(df_model$ID) < 20) {
    return(tibble(
      sex = as.character(sex_level),
      biomarker = str_remove(bm_base, "^nulisa_"),
      n_ids = n_distinct(df_model$ID),
      n_obs = NA_integer_,
      error = "Too few participants in sex stratum"
    ))
  }
  
  if ((needs_log(bm_base) && any(df_model$bm0_w <= 0, na.rm = TRUE))) {
    return(tibble(
      sex = as.character(sex_level),
      biomarker = str_remove(bm_base, "^nulisa_"),
      n_ids = n_distinct(df_model$ID),
      n_obs = NA_integer_,
      error = "Non-positive baseline values after winsorization; log not possible"
    ))
  }
  
  df_model <- df_model %>%
    mutate(
      bm0_log = transform_bm(bm0_w, bm_base),
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
    )
  df_long <- df_long %>%
    mutate(
      year = case_when(
        timepoint == "ztotal_composite_BL2"  ~ 0,
        timepoint == "ztotal_composite_12kk" ~ 1,
        timepoint == "ztotal_composite_24kk" ~ 2
      ),
      !!rlang::sym(var_group)  := factor(df_long[[var_group]]),
      apoe4_status = factor(apoe4_status),
      campid = factor(campid),
      bmi_bl2_z = as.numeric(scale(bmi_bl2))
    ) %>%
    filter(!is.na(ztotal_composite))
  
  if ("3" %in% levels(df_long$sub_sample_id)) {
    df_long$sub_sample_id <- relevel(df_long$sub_sample_id, ref = "3")
  }
    
  if ("1" %in% levels(df_long$sex)) {
    df_long$sex <- relevel(df_long$sex, ref = "1")
  }
  
  if ("0" %in% levels(df_long$apoe4_status)) {
    df_long$apoe4_status <- relevel(df_long$apoe4_status, ref = "0")
  }
  if ("101" %in% levels(df_long$campid)) {
    df_long$campid <- relevel(df_long$campid, ref = "101")
  }
  
  model <- tryCatch(
    lmer(
      paste0(
        "ztotal_composite ~ year * bm0_log_z * ", var_group,
        " + age_bl1 + apoe4_status + educ_years_BL1 + bmi_bl2_z + campid +",
        " (1 + year | ID)"
      ),
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
      sex = as.character(sex_level),
      biomarker = str_remove(bm_base, "^nulisa_"),
      n_ids = n_distinct(df_long$ID),
      n_obs = nrow(df_long),
      error = model$message
    ))
  }
  
  fixed <- tidy(model, effects = "fixed") %>%
    select(term, estimate, std.error, p.value)
  
  tibble(
    sex = as.character(sex_level),
    biomarker = str_remove(bm_base, "^nulisa_"),
    n_ids = n_distinct(df_long$ID),
    n_obs = nrow(df_long),
    
    `bm coeff` = get_term_value(fixed, "bm0_log_z", estimate),
    `bm std` = get_term_value(fixed, "bm0_log_z", std.error),
    `bm pvalue` = get_term_value(fixed, "bm0_log_z", p.value),
    
    `year coeff` = get_term_value(fixed, "year", estimate),
    `year std` = get_term_value(fixed, "year", std.error),
    `year pvalue` = get_term_value(fixed, "year", p.value),
    
    `group coeff` = get_term_value(fixed, ref_group, estimate),
    `group std` = get_term_value(fixed, ref_group, std.error),
    `group pvalue` = get_term_value(fixed, ref_group, p.value),
    
    `bm:year coeff` = get_term_value(fixed, "year:bm0_log_z", estimate),
    `bm:year std` = get_term_value(fixed, "year:bm0_log_z", std.error),
    `bm:year pvalue` = get_term_value(fixed, "year:bm0_log_z", p.value),
    
    `group:year coeff` = get_term_value(fixed, paste0("year:",ref_group), estimate),
    `group:year std` = get_term_value(fixed,   paste0("year:",ref_group), std.error),
    `group:year pvalue` = get_term_value(fixed,paste0("year:",ref_group), p.value),
    
    `group:bm coeff` = get_term_value(fixed, paste0("bm0_log_z:",ref_group), estimate),
    `group:bm std` = get_term_value(fixed, paste0("bm0_log_z:",ref_group), std.error),
    `group:bm pvalue` = get_term_value(fixed, paste0("bm0_log_z:",ref_group), p.value),
    
    `group:bm:year coeff` =
      get_term_value(fixed, paste0("year:bm0_log_z:",ref_group), estimate),
    `group:bm:year std` =
      get_term_value(fixed, paste0("year:bm0_log_z:",ref_group), std.error),
    `group:bm:year pvalue` =
      get_term_value(fixed, paste0("year:bm0_log_z:",ref_group), p.value),
    
    `age_0y coeff` = get_term_value(fixed, "age_bl1", estimate),
    `age_0y std` = get_term_value(fixed, "age_bl1", std.error),
    `age_0y pvalue` = get_term_value(fixed, "age_bl1", p.value),
    
    `bmi_0y coeff` = get_term_value(fixed, "bmi_bl2_z", estimate),
    `bmi_0y std` = get_term_value(fixed, "bmi_bl2_z", std.error),
    `bmi_0y pvalue` = get_term_value(fixed, "bmi_bl2_z", p.value),
    
    `apoe4_status coeff` = get_term_value(fixed, "apoe4_status1", estimate),
    `apoe4_status std` = get_term_value(fixed, "apoe4_status1", std.error),
    `apoe4_status pvalue` = get_term_value(fixed, "apoe4_status1", p.value),
    
    `educ_years_0y coeff` = get_term_value(fixed, "educ_years_BL1", estimate),
    `educ_years_0y std` = get_term_value(fixed, "educ_years_BL1", std.error),
    `educ_years_0y pvalue` = get_term_value(fixed, "educ_years_BL1", p.value),
    
    error = NA_character_
  )
}

# .............................................................................
# Run all biomarkers in each APOE stratum
# .............................................................................
sex_levels <- sort(unique(df_all$sex[!is.na(df_all$sex)]))

results_all <- map_dfr(
  sex_levels,
  function(sex_level) {
    map_dfr(
      biomarker_names,
      run_one_biomarker_stratum,
      data = df_all,
      sex_level = sex_level, 
      .progress = interactive()
    )
  }
)

# .............................................................................
# Add FDR within each APOE stratum
# .............................................................................
p_cols <- grep("pvalue$", names(results_all), value = TRUE)

results_all <- results_all %>%
  group_by(sex) %>%
  mutate(
    across(
      all_of(p_cols),
      ~ p.adjust(.x, method = "fdr"),
      .names = "{.col} FDR"
    )
  ) %>%
  ungroup()

# .............................................................................
# Round numeric columns
# .............................................................................
results_all <- results_all %>%
  mutate(across(where(is.numeric), ~ round(.x, 6)))

# .............................................................................
# Save results
# .............................................................................
write_csv(
  results_all,
  paste0(output_path,"Stratified_sex/sex_stratified_3way_mixed_models_all_biomarkers_log_z_bmi_adj.csv")
)


# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
# sex-stratified coefficient plot
## 3-way interaction: group × biomarker × year --------------------------------
# Different colors from total-cohort plots
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////


# .............................................................................
# File path
# .............................................................................
file_path <- paste0(output_path, "Stratified_sex/sex_stratified_3way_mixed_models_all_biomarkers_log_z_bmi_adj.csv")

# .............................................................................
# Proteins to exclude
# .............................................................................
exclude_proteins <- c(
  "pTau181",
  "pTau231",
  "pTau217",
  "AB38",
  "NEFL",
  "GFAP",
  "AB40",
  "lumipulse_ptau217"
)

# .............................................................................
# Read data
# .............................................................................
res <- read_csv(file_path, show_col_types = FALSE)

# .............................................................................
# Prepare plotting data
# .............................................................................
plot_df <- res %>%
  mutate(
    biomarker = str_remove(biomarker, "^nulisa_")
  ) %>%
  filter(!biomarker %in% exclude_proteins) %>%
  transmute(
    sex,
    biomarker,
    beta = `group:bm:year coeff`,
    se = `group:bm:year std`,
    p = `group:bm:year pvalue`,
    p_fdr = `group:bm:year pvalue FDR`
  ) %>%
  filter(!is.na(beta), !is.na(se), !is.na(p), p < 0.1) %>%
  mutate(
    ci_low = beta - 1.96 * se,
    ci_high = beta + 1.96 * se,
    star = if_else(!is.na(p_fdr) & p_fdr < 0.05, "*", ""),
    sig = case_when(
      p < 0.05 ~ "p < 0.05",
      p < 0.1 ~ "0.05 <= p < 0.1",
      TRUE ~ "NS"
    )
  )

# .............................................................................
# Keep top 20 per APOE stratum
# .............................................................................
plot_df <- plot_df %>%
  group_by(sex) %>%
  arrange(p, desc(abs(beta)), .by_group = TRUE) %>%
  slice_head(n = 20) %>%
  ungroup()

# .............................................................................
# Symmetric x-axis
# .............................................................................
max_val <- max(abs(c(plot_df$ci_low, plot_df$ci_high)), na.rm = TRUE)
max_val <- max_val * 1.15
x_limits <- c(-max_val, max_val)

# .............................................................................
# .............................................................................
# Plot function with p-value color categories
# .............................................................................
make_sex_plot <- function(df, sex_level, dark_color, light_color, panel_title) {
  
  df_sub <- df %>%
    filter(sex == sex_level) %>%
    arrange(beta) %>%
    mutate(
      biomarker = factor(biomarker, levels = biomarker),
      point_color = case_when(
        p < 0.05 ~ dark_color,
        p < 0.10 ~ light_color,
        TRUE ~ "grey70"
      )
    )
  
  ggplot(df_sub, aes(x = beta, y = biomarker)) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.5,
      color = "grey45"
    ) +
    geom_errorbarh(
      aes(xmin = ci_low, xmax = ci_high),
      height = 0.18,
      linewidth = 0.6,
      color = "grey45"
    ) +
    geom_point(
      aes(color = point_color),
      size = 3
    ) +
    scale_color_identity() +
    geom_text(
      aes(
        x = ci_high + 0.03 * max_val,
        label = star
      ),
      size = 5,
      fontface = "bold",
      hjust = 0
    ) +
    coord_cartesian(xlim = x_limits) +
    labs(
      title = panel_title,
      subtitle = "Group × biomarker × time",
      caption = "Dark points: raw p < 0.05; light points: 0.05 <= raw p < 0.10; * FDR < 0.05",
      x = expression(beta ~ "coefficient"),
      y = NULL
    ) +
    theme_classic(base_size = 12) +
    theme(
      legend.position = "none",
      plot.title = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(size = 10),
      axis.text.y = element_text(size = 11),
      axis.text.x = element_text(size = 11),
      axis.title.x = element_text(size = 11)
    )
}

# .............................................................................
# Create plots
# .............................................................................
sex_levels <- sort(unique(plot_df$sex))

p_male <- make_sex_plot(
  plot_df,
  sex_level = sex_levels[1],
  dark_color = "#017353",
  light_color = "#83DEC1",
  panel_title = "Male"
)

p_female <- make_sex_plot(
  plot_df,
  sex_level = sex_levels[2],
  dark_color = "#CC5500",
  light_color = "#EDA774",
  panel_title = "Female"
)

combined_plot <- p_male + p_female +
  plot_layout(ncol = 2)

print(combined_plot)



# .............................................................................
# Save
# .............................................................................
ggsave(
  filename = paste0(output_path,"Stratified_sex/sex_stratified_3way_forestplot_different_colours.png"),
  plot = combined_plot,
  width = 13,
  height = 8,
  dpi = 600
)

ggsave(
  filename = paste0(output_path,"Stratified_sex/sex_stratified_3way_forestplot_different_colours.svg"),
  plot = combined_plot,
  device = svglite::svglite,
  width = 13,
  height = 8
)
write_csv(
  plot_df,
  paste0(output_path,"Stratified_sex/sex_stratified_3way_forestplot_table.csv")
)







# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
## 4-way interaction with sex status -----------------------------------------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////

# 4-way interaction with sex status
# Mixed-effects models for all baseline NULISA biomarkers
# + lumipulse_ptau217
# Output: one row per biomarker, columns = beta / SE / raw p / FDR
#
# Specifications:
# - random intercept + random slope for year: (1 + year | ID)
# - campid reference = Helsinki (101)
# - group reference = control (0)
# - apoe reference = non carrier
# - winsorization at 1% and 99%, baseline and 2-year separately
# - log transformation AFTER winsorization
# - z-scaling AFTER log transformation
# - remove patients without baseline biomarker
# - remove patients without baseline cognition OR without at least one follow-up
# - remove patients without one of the cofactors


# .............................................................................
# File path
# .............................................................................

file_path <- paste0(processed_path,"dataset_original.csv")

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
  var_group,
  "age_bl1",
  "sex",
  "educ_years_BL1",
  "apoe4_status",
  "bmi_bl2",
  "campid",
  "ztotal_composite_BL2",
  "ztotal_composite_12kk",
  "ztotal_composite_24kk"
)


# .............................................................................
# One-biomarker model runner
# .............................................................................
run_one_biomarker <- function(bm_base, data) {
  
  bm0 <- paste0(bm_base, "_0")
  bm2 <- paste0(bm_base, "_2")

  
  cat("Running:", bm_base, "\n") 
  
  df_model <- data %>%
    select(all_of(c(base_vars, bm0, bm2))) %>%
    filter(
      !is.na(.data[[bm0]]),
      !is.na(ztotal_composite_BL2),
      (!is.na(ztotal_composite_12kk) | !is.na(ztotal_composite_24kk)),
      !is.na(age_bl1),
      !is.na(sex),
      !is.na(educ_years_BL1),
      !is.na(bmi_bl2),
      !is.na(apoe4_status),
      !is.na(campid),
      !is.na(.data[[var_group]])
    ) %>%
    mutate(
      bm0_w = winsorize(.data[[bm0]], p = 0.01),
      bm2_w = winsorize(.data[[bm2]], p = 0.01)
    )
  
  # Log requires positive values at both timepoints
  if ((needs_log(bm_base) && any(df_model$bm0_w <= 0, na.rm = TRUE)) ||
      (needs_log(bm_base) && any(df_model$bm2_w <= 0, na.rm = TRUE))) {
    return(tibble(
      biomarker = str_remove(bm_base, "^nulisa_"),
      n_ids = n_distinct(df_model$ID),
      n_obs = NA_integer_,
      error = "Non-positive values after winsorization; log not possible"
    ))
  }
  
  df_model <- df_model %>%
    mutate(
      bm0_log   = transform_bm(bm0_w, bm_base),
      bm2_log   = transform_bm(bm2_w, bm_base),
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
    ) 
  df_long <- df_long %>%
    mutate(
      year = case_when(
        timepoint == "ztotal_composite_BL2"  ~ 0,
        timepoint == "ztotal_composite_12kk" ~ 1,
        timepoint == "ztotal_composite_24kk" ~ 2
      ),      
      !!rlang::sym(var_group)  := factor(df_long[[var_group]]),
      apoe4_status = factor(apoe4_status),
      sex = factor(sex),
      campid = factor(campid),
      bmi_bl2_z = as.numeric(scale(bmi_bl2))
    ) %>%
    filter(!is.na(ztotal_composite))
  
  # Reference groups
  if ("3" %in% levels(df_long$sub_sample_id)) {
    df_long$sub_sample_id <- relevel(df_long$sub_sample_id, ref = "3")
  }
  if ("1" %in% levels(df_long$sex)) {
    df_long$sex <- relevel(df_long$sex, ref = "1")
  }
  
  if ("101" %in% levels(df_long$campid)) {
    df_long$campid <- relevel(df_long$campid, ref = "101")
  }
  
  if ("0" %in% levels(df_long$apoe4_status)) {
    df_long$apoe4_status <- relevel(df_long$apoe4_status, ref = "0")
  }
  
  model <- tryCatch(
    lmer(
      paste0(
        "ztotal_composite ~ year * bm0_log_z * ", var_group,
        "* sex + age_bl1 + apoe4_status + educ_years_BL1 + bmi_bl2_z + campid +",
        " (1 + year | ID)"
      ),
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
  
  
  control_male_slope <- get_linear_combo(
    model,
    terms = c("year:bm0_log_z")
  )
  
  intervention_male_slope <- get_linear_combo(
    model,
    terms = c(
      "year:bm0_log_z",
      paste0("year:bm0_log_z:",ref_group)
    )
  )
  
  control_female_slope <- get_linear_combo(
    model,
    terms = c(
      "year:bm0_log_z",
      "year:bm0_log_z:sex2"
    )
  )
  
  intervention_female_slope <- get_linear_combo(
    model,
    terms = c(
      "year:bm0_log_z",
      paste0( "year:bm0_log_z:",ref_group),
      "year:bm0_log_z:sex2",
      paste0( "year:bm0_log_z:",ref_group,":sex2")
    )
  )
  
  tibble(
    biomarker = str_remove(bm_base, "^nulisa_"),
    n_ids = n_distinct(df_long$ID),
    n_obs = nrow(df_long),
    
    `Intercept coeff` = get_term_value(fixed, "(Intercept)", estimate),
    `Intercept std`   = get_term_value(fixed, "(Intercept)", std.error),
    `Intercept pvalue`= get_term_value(fixed, "(Intercept)", p.value),
    
    `group coeff` = get_term_value(fixed, ref_group, estimate),
    `group std`   = get_term_value(fixed, ref_group, std.error),
    `group pvalue`= get_term_value(fixed, ref_group, p.value),
    
    `bm coeff` = get_term_value(fixed, "bm0_log_z", estimate),
    `bm std`   = get_term_value(fixed, "bm0_log_z", std.error),
    `bm pvalue`= get_term_value(fixed, "bm0_log_z", p.value),
    
    `group:bm coeff` = get_term_value(fixed, paste0("bm0_log_z:",ref_group), estimate),
    `group:bm std`   = get_term_value(fixed, paste0("bm0_log_z:",ref_group), std.error),
    `group:bm pvalue`= get_term_value(fixed, paste0("bm0_log_z:",ref_group), p.value),
    
    `year coeff` = get_term_value(fixed, "year", estimate),
    `year std`   = get_term_value(fixed, "year", std.error),
    `year pvalue`= get_term_value(fixed, "year", p.value),
    
    `group:year coeff` = get_term_value(fixed, paste0("year:",ref_group), estimate),
    `group:year std`   = get_term_value(fixed, paste0("year:",ref_group), std.error),
    `group:year pvalue`= get_term_value(fixed, paste0("year:",ref_group), p.value),
    
    `bm:year coeff` = get_term_value(fixed, "year:bm0_log_z", estimate),
    `bm:year std`   = get_term_value(fixed, "year:bm0_log_z", std.error),
    `bm:year pvalue`= get_term_value(fixed, "year:bm0_log_z", p.value),
    
    `group:bm:year coeff` =
      get_term_value(fixed, paste0("year:bm0_log_z:",ref_group), estimate),
    `group:bm:year std` =
      get_term_value(fixed, paste0("year:bm0_log_z:",ref_group), std.error),
    `group:bm:year pvalue` =
      get_term_value(fixed, paste0("year:bm0_log_z:",ref_group), p.value),
    
    `group:bm:year:sex coeff` =
      get_term_value(
        fixed,
        paste0("year:bm0_log_z:",ref_group,":sex2"),
        estimate
      ),
    
    `group:bm:year:sex std` =
      get_term_value(
        fixed,
        paste0("year:bm0_log_z:",ref_group,":sex2"),
        std.error
      ),
    
    `group:bm:year:sex pvalue` =
      get_term_value(
        fixed,
        paste0("year:bm0_log_z:",ref_group,":sex2"),
        p.value
      ),
    `control male bm:year coeff` = control_male_slope$estimate,
    `control male bm:year std` = control_male_slope$std.error,
    `control male bm:year pvalue` = control_male_slope$p.value,
    `control male bm:year ci_low` = control_male_slope$ci_low,
    `control male bm:year ci_high` = control_male_slope$ci_high,
    
    !!glue::glue("{interv_val_group} male bm:year coeff") := intervention_male_slope$estimate,
    !!glue::glue("{interv_val_group} male bm:year std") := intervention_male_slope$std.error,
    !!glue::glue("{interv_val_group} male bm:year pvalue") := intervention_male_slope$p.value,
    !!glue::glue("{interv_val_group} male bm:year ci_low" ):= intervention_male_slope$ci_low,
    !!glue::glue("{interv_val_group} male bm:year ci_high") := intervention_male_slope$ci_high,
    
    `control female bm:year coeff` = control_female_slope$estimate,
    `control female bm:year std` = control_female_slope$std.error,
    `control female bm:year pvalue` = control_female_slope$p.value,
    `control female bm:year ci_low` = control_female_slope$ci_low,
    `control female bm:year ci_high` = control_female_slope$ci_high,
    
    !!glue::glue("{interv_val_group} female bm:year coeff") := intervention_female_slope$estimate,
    !!glue::glue("{interv_val_group} female bm:year std") := intervention_female_slope$std.error,
    !!glue::glue("{interv_val_group} female bm:year pvalue") := intervention_female_slope$p.value,
    !!glue::glue("{interv_val_group} female bm:year ci_low") := intervention_female_slope$ci_low,
    !!glue::glue("{interv_val_group} female bm:year ci_high") := intervention_female_slope$ci_high,
    `age_0y coeff` = get_term_value(fixed, "age_bl1", estimate),
    `age_0y std`   = get_term_value(fixed, "age_bl1", std.error),
    `age_0y pvalue`= get_term_value(fixed, "age_bl1", p.value),
    
    `bmi_0y coeff` = get_term_value(fixed, "bmi_bl2_z", estimate),
    `bmi_0y std`   = get_term_value(fixed, "bmi_bl2_z", std.error),
    `bmi_0y pvalue`= get_term_value(fixed, "bmi_bl2_z", p.value),
    
    
    `apoe4_status coeff` = get_term_value(fixed, "apoe4_status1", estimate),
    `apoe4_status std`   = get_term_value(fixed, "apoe4_status1", std.error),
    `apoe4_status pvalue`= get_term_value(fixed, "apoe4_status1", p.value),
    
    `educ_years_0y coeff` =
      get_term_value(fixed, "educ_years_BL1", estimate),
    `educ_years_0y std` =
      get_term_value(fixed, "educ_years_BL1", std.error),
    `educ_years_0y pvalue` =
      get_term_value(fixed, "educ_years_BL1", p.value),
    
    error = NA_character_
  )
}
# .............................................................................
# Run all models
# .............................................................................
results_all <- map_dfr(biomarker_names, run_one_biomarker, data = df_all, .progress = interactive())

# .............................................................................
# Add FDR columns
# .............................................................................
p_cols <- grep("pvalue$", names(results_all), value = TRUE)

for (pcol in p_cols) {
  fdr_col <- paste0(pcol, " FDR")
  results_all[[fdr_col]] <- p.adjust(results_all[[pcol]], method = "fdr")
}

# .............................................................................
# Round numeric columns
# .............................................................................
results_all <- results_all %>%
  mutate(across(where(is.numeric), ~ round(.x, 6)))

# .............................................................................
# Save results
# .............................................................................
write_csv(
  results_all,
  paste0(output_path,"Stratified_sex/ztotal_composite_change_4way_mixed_model_baseline_bm_log_z_imputed_zeros_bmi_adj.csv")
)




# .............................................................................
### Top bm ----
# .............................................................................

results_all = read_csv(paste0(output_path,"Stratified_sex/ztotal_composite_change_4way_mixed_model_baseline_bm_log_z_imputed_zeros_bmi_adj.csv"))

top25_tables <- create_top25_tables(
  results = results_all,
  output_path = output_path,
  folder = "Stratified_sex",
  method = "group_bm_year_sex",
  p = "group:bm:year:sex pvalue"
)

top25_df <- top25_tables$top25_df
top25_df_low <- top25_tables$top25_df_low




# /////////////////////////////////////////////////////////
### Volcano plot  ----------------
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
  folder = "Stratified_sex",
  output_file_name = "volcano_ztotal_vs_time_vs_bbm_vs_group_vs_sex",
  title = "Volcano plot of 4-way interaction",
  results_filtered = results_filtered,
  top25_df = top25_df,
  p = "group:bm:year:sex pvalue",
  beta = "group:bm:year:sex coeff"
  
)



# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
## Coefficient plots for mixed model results ----------------------------------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////

# Coefficient plots for mixed model results
#
# Outputs:
#   1) Old two-panel plot:
#      A) baseline biomarker × time
#      B) group × baseline biomarker × time
#
#   2) New combined plot:
#      For each biomarker, two estimates are shown:
#      - Control: baseline biomarker × time
#      - Intervention group: control slope + group × biomarker × time
#
# New combined plot uses accurate SE/CI from model-derived
# linear combinations:
#   control bm:year coeff / ci_low / ci_high
#   intervention bm:year coeff / ci_low / ci_high



# .............................................................................
# File path
# .............................................................................
file_path <- paste0(output_path,"Stratified_sex/ztotal_composite_change_4way_mixed_model_baseline_bm_log_z_imputed_zeros_bmi_adj.csv")

# .............................................................................
# Proteins to exclude
# .............................................................................
exclude_proteins <- c(
  "pTau181",
  "pTau231",
  "pTau217",
  "AB38",
  "NEFL",
  "GFAP",
  "AB40",
  "lumipulse_ptau217"
)

# .............................................................................
# Read data
# .............................................................................
res <- read_csv(file_path, show_col_types = FALSE)

if (!"Protein" %in% names(res)) {
  names(res)[1] <- "Protein"
}

res <- res %>%
  mutate(
    Protein = str_remove(Protein, "^nulisa_")
  ) %>%
  filter(!Protein %in% exclude_proteins)

# ..............................................................................
# OLD FIGURE: separate coefficient plots
# .............................................................................

build_term_table <- function(data, beta_col, se_col, p_col,
                             top_n = 25, p_cutoff = 0.1) {
  data %>%
    transmute(
      Protein,
      beta = .data[[beta_col]],
      se   = .data[[se_col]],
      p    = .data[[p_col]]
    ) %>%
    filter(!is.na(beta), !is.na(se), !is.na(p)) %>%
    mutate(
      ci_low  = beta - 1.96 * se,
      ci_high = beta + 1.96 * se,
      sig = case_when(
        p < 0.05 ~ "p < 0.05",
        p < p_cutoff ~ "0.05 <= p < 0.1",
        TRUE ~ "NS"
      ),
      abs_beta = abs(beta)
    ) %>%
    filter(p < p_cutoff) %>%
    arrange(p, desc(abs_beta)) %>%
    slice_head(n = top_n) %>%
    arrange(beta) %>%
    mutate(Protein = factor(Protein, levels = Protein))
}

plot_bmyear <- build_term_table(
  data = res,
  beta_col = "bm:year coeff",
  se_col   = "bm:year std",
  p_col    = "bm:year pvalue",
  top_n    = 25,
  p_cutoff = 0.1
)

plot_3way <- build_term_table(
  data = res,
  beta_col = "group:bm:year coeff",
  se_col   = "group:bm:year std",
  p_col    = "group:bm:year pvalue",
  top_n    = 25,
  p_cutoff = 0.1
)

max_val_old <- max(
  abs(c(
    plot_bmyear$ci_low,
    plot_bmyear$ci_high,
    plot_3way$ci_low,
    plot_3way$ci_high
  )),
  na.rm = TRUE
)

x_limits_old <- c(-1.08 * max_val_old, 1.08 * max_val_old)

make_coef_plot <- function(df, panel_label, title, subtitle) {
  ggplot(df, aes(x = beta, y = Protein, color = sig)) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.5,
      color = "grey40"
    ) +
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
        "0.05 <= p < 0.1" = "#6baed6",
        "NS" = "grey70"
      ),
      drop = FALSE
    ) +
    coord_cartesian(xlim = x_limits_old) +
    labs(
      title = paste0(panel_label, "  ", title),
      subtitle = subtitle,
      x = "Beta coefficient",
      y = NULL,
      color = NULL
    ) +
    theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      plot.subtitle = element_text(size = 10),
      axis.text.y = element_text(size = 11),
      axis.text.x = element_text(size = 11),
      axis.title.x = element_text(size = 11),
      legend.position = "top",
      legend.text = element_text(size = 11)
    )
}

p1 <- make_coef_plot(
  df = plot_bmyear,
  panel_label = "A",
  title = "Baseline biomarker association with cognitive change",
  subtitle = paste0(
    "Top 25 biomarkers ranked by p value",
    "\nInteraction term: baseline biomarker x time"
  )
)

p2 <- make_coef_plot(
  df = plot_3way,
  panel_label = "B",
  title = paste0(test_group_val," modification of biomarker-associated cognitive change"),
  subtitle = paste0(
    "Top 25 biomarkers ranked by p value",
    paste0("\n",test_group_val," term: group x baseline biomarker x time")
  )
)

old_combined_plot <- p1 + p2 + patchwork::plot_layout(ncol = 2, guides = "collect") &
  theme(legend.position = "top")

print(old_combined_plot)

ggsave(
  filename = paste0(output_path,"Stratified_sex/mixed_models_top25_old_two_panel_forest.png"),
  plot = old_combined_plot,
  width = 13,
  height = 9,
  dpi = 600,
  bg = "white"
)

ggsave(
  filename = paste0(output_path,"Stratified_sex/mixed_models_top25_old_two_panel_forest.svg"),
  plot = old_combined_plot,
  device = svglite::svglite,
  width = 13,
  height = 9
)
ggsave(
  filename = paste0(output_path,"Stratified_sex/mixed_models_top25_old_two_panel_forest.pdf"),
  plot = old_combined_plot,
  width = 13,
  height = 9,
  bg = "white"
)


# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
## 4-WAY FIGURE WITH TWO PANELS — REFINED ----------------------------------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////

# Panel A:
#   Formal 4-way interaction effect:
#   group x biomarker x time x APOE4
#
# Panel B:
#   Exact subgroup-specific slopes:
#   Control / Intervention groupx Male / Female
#
# Main figure includes only biomarkers with raw 4-way p < 0.05.
# FDR-significant 4-way interactions are marked with **.
# Raw p < 0.05 are marked with *.

required_4way_cols <- c(
  "control male bm:year coeff",
  "control male bm:year pvalue",
  "control male bm:year ci_low",
  "control male bm:year ci_high",
  paste0(interv_val_group, " male bm:year coeff"),
  paste0(interv_val_group, " male bm:year pvalue"),
  paste0(interv_val_group, " male bm:year ci_low"),
  paste0(interv_val_group, " male bm:year ci_high"),
  "control female bm:year coeff",
  "control female bm:year pvalue",
  "control female bm:year ci_low",
  "control female bm:year ci_high",
  paste0(interv_val_group, " female bm:year coeff"),
  paste0(interv_val_group, " female bm:year pvalue"),
  paste0(interv_val_group, " female bm:year ci_low"),
  paste0(interv_val_group, " female bm:year ci_high"),
  "group:bm:year:sex coeff",
  "group:bm:year:sex std",
  "group:bm:year:sex pvalue"
)

missing_4way_cols <- setdiff(required_4way_cols, names(res))

if (length(missing_4way_cols) > 0) {
  warning(
    paste0(
      "4-way plot skipped because these columns are missing:\n",
      paste(missing_4way_cols, collapse = "\n"),
      "\n\nRe-run the 4-way mixed model script after adding exact APOE/group-specific linear combinations."
    )
  )
} else {
  
  fourway_fdr_col <- "group:bm:year:sex pvalue FDR"
  has_fourway_fdr <- fourway_fdr_col %in% names(res)
  
  txt_female <- paste0("Stronger ",test_group_val, " modification in females")
  txt_male <- paste0("Stronger ",test_group_val, " modification in males")
  
  
  if (!has_fourway_fdr) {
    message("No 4-way FDR column found. Only raw p-value significance will be marked.")
  }
  
  plot_4way_wide <- res %>%
    dplyr::transmute(
      Protein,
      fourway_beta = `group:bm:year:sex coeff`,
      fourway_se   = `group:bm:year:sex std`,
      fourway_p    = `group:bm:year:sex pvalue`,
      fourway_fdr  = if (has_fourway_fdr) res[[fourway_fdr_col]] else NA_real_,
      
      control_male_beta    = `control male bm:year coeff`,
      control_male_p       = `control male bm:year pvalue`,
      control_male_ci_low  = `control male bm:year ci_low`,
      control_male_ci_high = `control male bm:year ci_high`,
      
      intervention_male_beta    = res[[paste0(interv_val_group, " male bm:year coeff")]],
      intervention_male_p       = res[[paste0(interv_val_group, " male bm:year pvalue")]],
      intervention_male_ci_low  = res[[paste0(interv_val_group, " male bm:year ci_low")]],
      intervention_male_ci_high = res[[paste0(interv_val_group, " male bm:year ci_high")]],
      
      control_female_beta    = `control female bm:year coeff`,
      control_female_p       = `control female bm:year pvalue`,
      control_female_ci_low  = `control female bm:year ci_low`,
      control_female_ci_high = `control female bm:year ci_high`,
      
      intervention_female_beta    = res[[paste0(interv_val_group, " female bm:year coeff")]],
      intervention_female_p       = res[[paste0(interv_val_group, " female bm:year pvalue")]],
      intervention_female_ci_low  = res[[paste0(interv_val_group, " female bm:year ci_low")]],
      intervention_female_ci_high = res[[paste0(interv_val_group, " female bm:year ci_high")]]
      
    ) %>%
    dplyr::filter(
      !is.na(fourway_beta),
      !is.na(fourway_se),
      !is.na(fourway_p)
    ) %>%
    dplyr::mutate(
      fourway_ci_low  = fourway_beta - 1.96 * fourway_se,
      fourway_ci_high = fourway_beta + 1.96 * fourway_se,
      
      sex_difference = abs(
        intervention_female_beta - control_female_beta
      ) - abs(
        intervention_male_beta - control_male_beta
      ),
      
      direction_block = dplyr::case_when(
        sex_difference >= 0 ~ txt_female,
        sex_difference < 0  ~ txt_male
      ),
      
      fourway_marker = dplyr::case_when(
        !is.na(fourway_fdr) & fourway_fdr < 0.05 ~ "**",
        fourway_p < 0.05 ~ "*",
        TRUE ~ ""
      )
    ) %>%
    dplyr::filter(fourway_p < 0.05) %>%
    dplyr::arrange(dplyr::desc(abs(fourway_beta)))
  
  if (nrow(plot_4way_wide) == 0) {
    stop("No biomarkers with raw 4-way p < 0.05 after filtering.")
  }
  
  protein_order_4way <- plot_4way_wide %>%
    dplyr::arrange(fourway_beta) %>%
    dplyr::pull(Protein)
  
  plot_4way_wide <- plot_4way_wide %>%
    dplyr::mutate(
      Protein = factor(Protein, levels = protein_order_4way)
    )
  
  p_A <- ggplot(plot_4way_wide, aes(x = fourway_beta, y = Protein)) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.55,
      color = "grey45"
    ) +
    geom_errorbarh(
      aes(xmin = fourway_ci_low, xmax = fourway_ci_high),
      height = 0.22,
      linewidth = 0.90,
      color = "grey35"
    ) +
    geom_point(
      aes(color = direction_block),
      size = 4.3,
      shape = 18
    ) +
    geom_text(
      aes(
        x = dplyr::if_else(
          fourway_beta >= 0,
          fourway_ci_high + 0.004,
          fourway_ci_low - 0.004
        ),
        label = fourway_marker
      ),
      size = 4.3,
      fontface = "bold",
      show.legend = FALSE
    ) +
    scale_color_manual(
      values = setNames(
        c(color_female, color_male),
        c(
          paste0("Stronger ", test_group_val, " modification in females"),
          paste0("Stronger ", test_group_val, " modification in males")
        )
      )
    )+
    
    coord_cartesian(
      xlim = c(-0.14, 0.14),
      clip = "off"
    ) +
    labs(
      title = paste0("A. Sex-dependent ",test_group_val," modification"),
      x = "4-way beta coefficient",
      y = NULL,
      color = NULL
    ) +
    guides(
      color = guide_legend(nrow = 2, byrow = TRUE)
    ) +
    
    theme_classic(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 10.5, color = "grey35"),
      axis.text.y = element_text(size = 11),
      axis.text.x = element_text(size = 11),
      axis.title.x = element_text(size = 11),
      legend.position = "top",
      legend.justification = "left",
      legend.direction = "vertical",
      legend.margin = margin(0, 0, -3, 0),
      legend.box.spacing = unit(0.2, "lines"),
      legend.title.align = 0,
      legend.text.align = 0,
      legend.text = element_text(size = 8),
      plot.margin = margin(10, 25, 10, 10)
    )
  
  
  plot_4way_long <- plot_4way_wide %>%
    tidyr::pivot_longer(
      cols = c(
        control_male_beta,
        control_male_p,
        control_male_ci_low,
        control_male_ci_high,
        intervention_male_beta,
        intervention_male_p,
        intervention_male_ci_low,
        intervention_male_ci_high,
        control_female_beta,
        control_female_p,
        control_female_ci_low,
        control_female_ci_high,
        intervention_female_beta,
        intervention_female_p,
        intervention_female_ci_low,
        intervention_female_ci_high
      ),
      names_to = c("TrialGroup", "sex", ".value"),
      names_pattern = "(control|intervention)_(male|female)_(beta|p|ci_low|ci_high)"
    ) %>%
    dplyr::mutate(
      TrialGroup = dplyr::recode(
        TrialGroup,
        control = "Control",
        intervention = test_group_val
      ),
      sex = dplyr::recode(
        sex,
        male = "Male",
        female = "Female"
      ),
      TrialGroup = factor(TrialGroup, levels = c("Control",test_group_val)),
      sex = factor(sex, levels = c("Male", "Female")),
      
      subgroup_marker = dplyr::if_else(p < 0.05, "*", ""),
      label_x = dplyr::case_when(
        beta >= 0 ~ ci_high + 0.0015,
        beta < 0  ~ ci_low  - 0.0015
      )
    )
  plot_4way_long$TrialGroup <- factor(
    plot_4way_long$TrialGroup,
    levels = c("Control", test_group_val)
  )
  
  plot_4way_long$sex <- factor(
    plot_4way_long$sex,
    levels = c("Male", "Female")
  )
  color_val = c(
    "Control" = color_control,
    "Intervention" = color_interv
  )
  
  # Create a new variable to define the order of groups
  plot_4way_long$group_order <- factor(
    paste(plot_4way_long$sex, plot_4way_long$TrialGroup, sep = "_"),
    levels = c(
      "Male_Control",
      paste0("Male_",test_group_val),
      "Female_Control",
      paste0("Female_",test_group_val)
    )
  )
  
  p_B <- ggplot(
    plot_4way_long,
    aes(
      x = beta,
      y = Protein,
      color = TrialGroup,
      shape = sex,
      group = group_order  # Ensure grouping follows the desired order
    )
  ) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.55,
      color = "grey45"
    ) +
    geom_errorbarh(
      aes(xmin = ci_low, xmax = ci_high, linetype = sex),
      height = 0.22,
      linewidth = 0.90,
      position = position_dodge(width = 0.75)
    ) +
    geom_point(
      size = 2.7,
      position = position_dodge(width = 0.75)
    ) +
    geom_text(
      aes(x = label_x, label = subgroup_marker),
      position = position_dodge(width = 0.75),
      size = 4.3,
      fontface = "bold",
      show.legend = FALSE
    ) +
    scale_color_manual(
      values = color_val
    ) +
    scale_shape_manual(
      values = c(
        "Male" = 16,
        "Female" = 17
      )
    ) +
    scale_linetype_manual(
      values = c(
        "Male" = "dashed",  # Dotted lines for non-carriers
        "Female" = "solid"         # Solid lines for carriers
      )
    ) +
    coord_cartesian(
      xlim = c(-0.08, 0.08),
      clip = "off"
    ) +
    labs(
      title = "B. Sex- and group-specific biomarker slopes",
      x = "Estimated beta coefficient for cognitive change",
      y = NULL,
      color = "Trial group",
      shape = "Sex",
      linetype = "Sex"  # Add legend for linetype
    ) +
    theme_classic(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 10.5, color = "grey35"),
      axis.text.y = element_text( size = 11),
      axis.ticks.y = element_blank(),
      axis.text.x = element_text(size = 11),
      axis.title.x = element_text(size = 11),
      legend.position = "top",
      legend.justification = "left",
      legend.margin = margin(0, 0, -3, 0),
      legend.box.spacing = unit(0.2, "lines"),
      legend.title.align = 0,
      legend.text.align = 0,
      legend.text = element_text(size = 8),
      legend.title = element_text(size = 9),
      plot.margin = margin(10, 25, 10, 5),
      #legend.box = "vertical",  # Stack legends vertically
      #legend.spacing.x = unit(0,"cm"),
      legend.key.size = unit(1.5, "cm")  # Increase the size of the legend key area
      
    ) 
  
  combined_4way <- p_A + p_B +
    patchwork::plot_layout(
      ncol = 2,
      widths = c(0.95, 1.35)
    ) &
    theme(
      panel.spacing = unit(1.5, "lines")
    )
  
  print(combined_4way)
  
  fig_height_4way <- max(9, 0.55 * nrow(plot_4way_wide))
  
  ggsave(
    filename = paste0(output_path,"Stratified_sex/mixed_models_4way_sex_two_panel_interaction_and_slopes_p005_refined.png"),
    plot = combined_4way,
    width = 15,
    height = fig_height_4way,
    dpi = 600,
    bg = "white"
  )
  
  ggsave(
    filename = paste0(output_path,"Stratified_sex/mixed_models_4way_sex_two_panel_interaction_and_slopes_p005_refined.svg"),
    plot = combined_4way,
    device = svglite::svglite,
    width = 15,
    height = fig_height_4way
  )
  ggsave(
    filename = paste0(output_path,"Stratified_sex/mixed_models_4way_sex_two_panel_interaction_and_slopes_p005_refined.pdf"),
    plot = combined_4way,
    width = 15,
    height = fig_height_4way,
    bg = "white"
  )
  
  readr::write_csv(
    plot_4way_wide,
    paste0(output_path,"Stratified_sex/mixed_models_4way_sex_panelA_formal_interaction_p005.csv")
  )
  
  readr::write_csv(
    plot_4way_long,
    paste0(output_path,"Stratified_sex/mixed_models_4way_sex_panelB_subgroup_slopes_p005.csv")
  )
}

# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
## SUPPLEMENTARY TABLE FOR 4-WAY APOE INTERACTION FIGURE ----------------------------------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
#
# Includes:
# - Formal 4-way interaction statistics
# - FDR correction
# - Exact subgroup-specific slopes
# - Direction labels
# -  Includes biomarkers with nominal 4-way p < 0.10
# Output:
#   Supplementary_Table_APOE4_4way_interactions_p010.csv


# .............................................................................
# File path
# .............................................................................
file_path <- paste0(output_path,"Stratified_sex/ztotal_composite_change_4way_mixed_model_baseline_bm_log_z_imputed_zeros_bmi_adj.csv")

# .............................................................................
# Read results
# .............................................................................
res <- read_csv(file_path, show_col_types = FALSE)

if (!"Protein" %in% names(res)) {
  names(res)[1] <- "Protein"
}

res <- res %>%
  mutate(
    Protein = str_remove(Protein, "^nulisa_")
  )

# .............................................................................
# Detect optional FDR column
# .............................................................................
fourway_fdr_col <- "group:bm:year:sex pvalue FDR"
has_fdr <- fourway_fdr_col %in% names(res)

# .............................................................................
# Build supplementary table
# .............................................................................
supp_table <- res %>%
  transmute(
    Protein,
    
    fourway_beta = `group:bm:year:sex coeff`,
    fourway_se   = `group:bm:year:sex std`,
    fourway_p    = `group:bm:year:sex pvalue`,
    fourway_fdr  = if (has_fdr) res[[fourway_fdr_col]] else NA_real_,
    
    fourway_ci_low  = fourway_beta - 1.96 * fourway_se,
    fourway_ci_high = fourway_beta + 1.96 * fourway_se,
    abs_fourway_beta = abs(fourway_beta),
    
    control_male_beta = `control male bm:year coeff`,
    control_male_p    = `control male bm:year pvalue`,
    
    intervention_male_beta = res[[paste0(interv_val_group, " male bm:year coeff")]],
    intervention_male_p    = res[[paste0(interv_val_group, " male bm:year pvalue")]],
    
    control_female_beta = `control female bm:year coeff`,
    control_female_p    = `control female bm:year pvalue`,
    
    intervention_female_beta = res[[paste0(interv_val_group, " female bm:year coeff")]],
    intervention_female_p    = res[[paste0(interv_val_group, " female bm:year pvalue")]]
  ) %>%
  mutate(
    apoe_difference = abs(
      intervention_female_beta - control_female_beta
    ) - abs(
      intervention_male_beta - control_male_beta
    ),
    
    interaction_direction = case_when(
      apoe_difference > 0 ~
        paste0("Stronger ",test_group_val,"-related biomarker association in Females"),
      apoe_difference < 0 ~
        paste0("Stronger ",test_group_val,"-related biomarker association in Males"),
      TRUE ~
        "No directional difference"
    ),
    
    significance_label = case_when(
      !is.na(fourway_fdr) & fourway_fdr < 0.05 ~ "FDR significant",
      fourway_p < 0.05 ~ "Nominal p < 0.05",
      fourway_p < 0.10 ~ "Nominal p 0.05-0.10",
      TRUE ~ "Not significant"
    )
  ) %>%
  filter(fourway_p < 0.10) %>%
  arrange(fourway_p)

# .............................................................................
# Round numeric columns
# .............................................................................
supp_table <- supp_table %>%
  mutate(
    across(where(is.numeric), ~ round(.x, 4))
  )

# .............................................................................
# Save table
# .............................................................................
write_csv(
  supp_table,
  paste0(output_path,"Stratified_sex/Supplementary_Table_SEX_4way_interactions_p010.csv")
)


















# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
# Correlations of the top proteins with the "core AD" biomarkers ---------------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////

threshold = 0.4
option_top <- "group_bm_year_apoe" #"group_bm_year" or ""bm_change_linear" "group_bm_year_apoe" "group_bm_year_sex"
pval_top <- 0.05


# .............................................................................
# Files path
# .............................................................................

if (option_top == "group_bm_year_apoe"){
  folder <- "Stratified"
}else if (option_top == "group_bm_year_sex"){
  folder <- "Stratified_sex"
  
}else{
  folder <- "Longitudinal"
  
}

top25_path <- paste0(output_path,folder,"/top25_",option_top,"_pval_",pval_top,"_excluding_core.csv")

data_path <- paste0(processed_path,"dataset_original.csv")
# .............................................................................
# Read data
# .............................................................................

top25_df <- read_csv(top25_path, show_col_types = FALSE)
df <- read_csv(data_path, show_col_types = FALSE)


df["nulisa_AB42/40_0"] <- df["nulisa_AB42_0"] / df["nulisa_AB40_0"]
df["nulisa_AB42/40_2"] <- df["nulisa_AB42_2"] / df["nulisa_AB40_2"]

df["ztotal_cognition_baseline"] <- df["ztotal_composite_BL2"]
df["ztotal_cognition_change"] <- df["ztotal_composite_24kk"] - df["ztotal_composite_BL2"]

# .............................................................................
# Core proteins
# .............................................................................

 if (any(startsWith(top25_df$Protein, "nulisa_"))){
core_proteins <- c(
  paste0(
    "nulisa_",
    c(
      "pTau181",
      "pTau231",
      "pTau217",
      "AB38",
      "NEFL",
      "GFAP",
      "AB42/40",
      "AB40"
    ),
    "_0"
  ),
  "lumipulse_ptau217_0"#,
#  "apoe4_status"
)}else{
  core_proteins <- c(
      "AB38",
      "AB42/40",
      "AB40",
      "NEFL",
      "GFAP",
      "pTau181",
      "pTau231",
      "pTau217",
  "lumipulse_ptau217"#,
  #"apoe4_status"
)
  
}
# .............................................................................
# Get protein names and extract columns
# .............................................................................

filtered_proteins <- top25_df %>%
  dplyr::filter(!Protein %in% core_proteins) %>%
  dplyr::pull(Protein) %>%
  paste0("nulisa_", .,"_0")

core_proteins <- c(
    paste0(
      "nulisa_",
      c(
        "AB42/40",
        "GFAP",
        "NEFL",
        "pTau181",
        "pTau217",
        "pTau231"
        #"AB38",
        #"AB40"
      ),
      "_0"
    ),
    "lumipulse_ptau217_0"#,
    #"apoe4_status"
  )

if (option_top == "bm_change_linear"){
    
  core_proteins <- c(
    core_proteins,
    "ztotal_cognition_baseline",
    "ztotal_cognition_change"
  )

  
}


all_proteins <- c(filtered_proteins, core_proteins)


df_corr <- df[, intersect(all_proteins, names(df))]


cor_pmat <- function(df) {
  vars <- colnames(df)
  p_mat <- matrix(NA, ncol = length(vars), nrow = length(vars))
  colnames(p_mat) <- vars
  rownames(p_mat) <- vars
  
  for (i in 1:length(vars)) {
    for (j in 1:length(vars)) {
      test <- suppressWarnings(
        cor.test(df[[i]], df[[j]], method = "spearman", exact = FALSE)
      )
      p_mat[i, j] <- test$p.value
    }
  }
  
  p_mat
}

cor_mat <- cor(df_corr, method = "spearman", use = "pairwise.complete.obs")
p_mat   <- cor_pmat(df_corr)

cor_cross <- cor_mat[filtered_proteins, core_proteins, drop = FALSE]
p_cross   <- p_mat[filtered_proteins, core_proteins, drop = FALSE]
cor_long <- as.data.frame(cor_cross) %>%
  tibble::rownames_to_column("FilteredProtein") %>%
  tidyr::pivot_longer(
    cols = -FilteredProtein,
    names_to = "CoreProtein",
    values_to = "rho"
  )

p_long <- as.data.frame(p_cross) %>%
  tibble::rownames_to_column("FilteredProtein") %>%
  tidyr::pivot_longer(
    cols = -FilteredProtein,
    names_to = "CoreProtein",
    values_to = "pvalue"
  )

cor_long <- dplyr::left_join(cor_long, p_long,
                             by = c("FilteredProtein", "CoreProtein"))




cor_long <- cor_long %>%
  dplyr::mutate(
    sig = dplyr::case_when(
      pvalue < 0.01 ~ "***",
      pvalue < 0.05 ~ "**",
      pvalue < 0.1  ~ "*",
      TRUE ~ ""
    ),
    
    label = dplyr::case_when(
      abs(rho) > threshold ~ paste0(round(rho, 2), sig),
      TRUE ~ ""
    )
  )
if (option_top == "bm_change_linear"){
  title_option <- paste0("bm change by ", group_name)
}else if (option_top == "group_bm_year"){
  title_option <- paste0("cognitive change by bm x ", group_name)
}else{
  title_option <- "cognitive change by bm"
  
}

title_fig <- paste0("Spearman correlations between top proteins (",title_option,"), \n core proteins and cognition")


cor_long$CoreProtein <- factor(
  cor_long$CoreProtein,
  levels = all_proteins
)


cor_long <- rename_proteins(cor_long, "CoreProtein")
cor_long <- rename_proteins(cor_long, "FilteredProtein")

p_corr <- ggplot(cor_long, aes(x = CoreProtein, y = FilteredProtein, fill = rho)) +
  geom_tile(color = "white") +
  
  geom_text(aes(label = label), size = 3) +
  
  scale_fill_gradient2(
    low = "#2C7BB6",
    mid = "white",
    high = "#D7191C",
    midpoint = 0,
    limits = c(-1, 1),
    name = "Spearman\nrho"
  ) +
  
  labs(
    title = title_fig,
    x = NULL,
    y = NULL
  ) +
  
  theme_classic(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text = element_text(size = 9)
  )

if (option_top == "group_bm_year_apoe"){
  folder <- "Stratified"
}else{
  folder <- "cross-sectional"
  
}


ggsave(
  filename = paste0(output_path,folder,"/protein_correlation_heatmap_",option_top,"pval_",pval_top,".png"),
  plot = p_corr,
  width = 10,
  height = 8,
  dpi = 600
)

ggsave(
  filename = paste0(output_path,folder,"/protein_correlation_heatmap_",option_top,"pval_",pval_top,".svg"),
  plot = p_corr,
  device = svglite::svglite,
  width = 10,
  height = 8
)


# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
# Heatmap of standardized regression coefficients (proteins × Age/Sex/APOE) ---------------
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////


# .............................................................................
# File path
# .............................................................................
file_path <- paste0(processed_path,"dataset_original.csv")
# .............................................................................
# Load data
# .............................................................................
df_all <- read_csv(file_path, show_col_types = FALSE)


# .............................................................................
# Identify biomarkers
# .............................................................................
nulisa_0_cols <- names(df_all)[grepl("^nulisa_.*_0$", names(df_all))]

exclude_proteins <- c(
  "nulisa_pTau181_0",
  "nulisa_pTau231_0",
  "nulisa_pTau217_0",
  "nulisa_AB38_0",
  "nulisa_NEFL_0",
  "nulisa_GFAP_0",
  "nulisa_AB40_0"
)


biomarker_names <- setdiff(nulisa_0_cols, exclude_proteins)
biomarker_names <- drop_excluded(biomarker_names)


## covariates
covariates <- c("age_bl1","sex","apoe4_status")

list_variables <- c(covariates, c("sub_sample_id"),biomarker_names)



for (g in c("all",'intervention','control')){
  if (g == "intervention"){
    df <- df_all %>% filter(sub_sample_id == 2) %>% select(list_variables) 
  }else if(g == "control"){
    df <- df_all %>% filter(sub_sample_id == 3) %>% select(list_variables) 
  }else{
    df <- df_all %>% select(list_variables) 
  }
  results <- list()
  
  pb <- txtProgressBar(min = 0, max = length(biomarker_names), style = 3)
  
  for (i in seq_along(biomarker_names)) {
    prot <- biomarker_names[i]
    
    
    dat <- df %>%
      select(all_of(c(prot, covariates))) %>%
      drop_na()
    
    if (nrow(dat) < 20)
      next
    
    fit <- lm(
      reformulate(covariates, response = prot),
      data = dat
    )
    
    pars <- model_parameters(
      fit,
      standardize = "refit"
    )
    
    pars <- pars %>%
      filter(Parameter %in% covariates) %>%
      transmute(
        Protein = prot,
        Variable = Parameter,
        Beta = Coefficient ,
        SE = SE,
        p = p
      )
    
    results[[prot]] <- pars
    
    setTxtProgressBar(pb, i)
    
  }
  
  results <- bind_rows(results)
  
  ## FDR correction within each predictor
  results <- results %>%
    group_by(Variable) %>%
    mutate(FDR = p.adjust(p, method = "BH")) %>%
    ungroup()
  
  write.csv(
    results,
    paste0(output_path,"cross-sectional/Protein_associations_Age_Sex_APOE_",g,".csv"),
    row.names = FALSE
  )
  
  ## Keep proteins with at least one significant association
  results_sig <- results %>%
    group_by(Protein) %>%
    filter(any(FDR < 0.05)) %>%
    ungroup()
  
  heat <- results_sig %>%
    select(Protein, Variable, Beta) %>%
    pivot_wider(
      names_from = Variable,
      values_from = Beta
    )
  
  heat_mat <- heat %>%
    column_to_rownames("Protein") %>%
    as.matrix()
  sig <- results_sig %>%
    mutate(
      stars = case_when(
        FDR < 0.001 ~ "***",
        FDR < 0.01  ~ "**",
        FDR < 0.05  ~ "*",
        TRUE        ~ ""
      ),
      label = case_when(
        abs(Beta) > 0.2 & stars != "" ~ sprintf("%.2f %s", Beta, stars),
        abs(Beta) > 0.2 & stars == "" ~ sprintf("%.2f", Beta),
        stars != "" ~ stars,
        TRUE ~ ""
      )
    ) %>%
    select(Protein, Variable, label) %>%
    pivot_wider(
      names_from = Variable,
      values_from = label
    )
  
  sig_mat <- sig %>%
    column_to_rownames("Protein") %>%
    as.matrix()
  
  cols <- colorRampPalette(c("#2166AC", "white", "#B2182B"))(101)
  
  
  
  
  
  rownames(heat_mat) <- rownames(heat_mat) |>
    gsub("^nulisa_", "", x = _) |>
    gsub("_0$", "", x = _)
  
  rownames(sig_mat) <- rownames(heat_mat)
  
  p <- pheatmap(
    heat_mat,
    color = cols,
    breaks = seq(-1, 1, length.out = 102),
    display_numbers = sig_mat,
    number_color = "black",
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    border_color = NA,
    
    fontsize_row = 7,
    fontsize_col = 9,
    fontsize_number = 9,
    angle_col = 0,      # or 90, 270, 315
  
    cellheight = 8,   # increase vertical spacing between proteins
    cellwidth = 60,
    main = "Association of baseline proteins with age, sex and APOE ε4"
    )
  
  fig_height_4way <- max(9, 0.55 * nrow(p))
  
  
  ggsave(
    filename = paste0(output_path,"cross-sectional/heatmap_age_sex_apoe_proteins_",g,".png"),
    plot = p,
    width = 15,
    height = fig_height_4way,
    dpi = 600,
    bg = "white"
  )
  
  
  ggsave(
    filename = paste0(output_path,"cross-sectional/heatmap_age_sex_apoe_proteins_",g,".svg"),
    plot = p,
    device = svglite::svglite,
    width = 15,
    height = fig_height_4way
  )
  
  
  ggsave(
    filename = paste0(output_path,"cross-sectional/heatmap_age_sex_apoe_proteins_",g,".pdf"),
    plot = p,
    width = 15,
    height = fig_height_4way,
    bg = "white"
  )
}

