# /////////////////////////////////////////////////////////////////////////////
# Cognitive responder analysis among high-adherence FINGER participants
## APOE4 and core AD biomarkers excluded from all protein models.
# /////////////////////////////////////////////////////////////////////////////
#
# Purpose:
#   Examine the distribution of 2-year cognitive change among participants
#   with high adherence to the FINGER multidomain intervention.
#
# Cognitive change:
#   ztotal_composite_24kk - ztotal_composite_BL2
#
# Responder definition:
#   Responder     = 2-year cognitive change > 0
#   Non-responder = 2-year cognitive change <= 0
#
# High adherence:
#   multiparti %in% c(7, 8)
#
# Outputs:
#   1. Descriptive statistics for cognitive change
#   2. Responder/non-responder counts and percentages
#   3. Baseline cognition by responder status
#   4. Histogram of cognitive change
#   5. Histogram colored by responder status
#   6. Density plot
#   7. Dataset containing high-adherence participants and responder status
#
# /////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////
# Packages
# /////////////////////////////////////////////////////////////////////////////

packages <- c(
  "dplyr",
  "ggplot2",
  "readr", 
  "tidyr",
  "purrr",
  "stringr",
  "broom",
  "gtsummary",
  "ggrepel",
  "forcats"
  
)



for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}


# /////////////////////////////////////////////////////////////////////////////
# User and paths
# /////////////////////////////////////////////////////////////////////////////


folder_path <- "../data/"
original_path <- paste0(folder_path,"original/")
processed_path <- paste0(folder_path,"processed/")
output_path <- paste0(folder_path,"outputs_R/Responder_analysis")

if (!dir.exists(output_path)) {
  dir.create(output_path, recursive = TRUE)
}


# /////////////////////////////////////////////////////////////////////////////
# Exclusions
# /////////////////////////////////////////////////////////////////////////////
# Excluded from ALL protein models (and therefore from FDR):
# - core AD biomarkers (already excluded from the figures)
# - NULISA APOE4 (reflects the APOE e4 allele)
EXCLUDE_FROM_MODELS <- c(
  "pTau181", "pTau231", "pTau217",
  "AB38", "AB40", "AB42",
  "NEFL", "GFAP",
  "APOE4",
  "lumipulse_ptau217"
)

drop_excluded <- function(bm_names) {
  short <- sub("_0$", "", sub("^nulisa_", "", bm_names))
  bm_names[!short %in% EXCLUDE_FROM_MODELS]
}



# /////////////////////////////////////////////////////////////////////////////
# Load main dataset
# /////////////////////////////////////////////////////////////////////////////

file_path <- paste0(
  processed_path,
  "dataset_original.csv"
)

df_first <- read_csv(
  file_path,
  show_col_types = FALSE
)


# /////////////////////////////////////////////////////////////////////////////
# Create adherence groups
# /////////////////////////////////////////////////////////////////////////////

df_all <- df_all %>%
  mutate(
    adherence_class = case_when(
      
      # Control group
      sub_sample_id == 3 ~ "control",
      
      # High-adherence intervention group
      multiparti %in% c(7, 8) ~ "high",
      
      # Lower/medium adherence intervention group
      multiparti %in% 1:6 ~ "medium",
      
      TRUE ~ NA_character_
    )
  )


# Check adherence groups
print(
  table(
    df_all$adherence_class,
    useNA = "ifany"
  )
)


# /////////////////////////////////////////////////////////////////////////////
# Select high-adherence participants with complete cognitive data
# /////////////////////////////////////////////////////////////////////////////

df_high_adh <- df_all %>%
  filter(
    adherence_class == "high",
    !is.na(ztotal_composite_BL2),
    !is.na(ztotal_composite_24kk)
  ) %>%
  mutate(
    
    # 2-year cognitive change
    cog_change_2y =
      ztotal_composite_24kk -
      ztotal_composite_BL2,
    
    # Responder definition
    responder = if_else(
      cog_change_2y > 0,
      "Responder",
      "Non-responder"
    ),
    
    responder = factor(
      responder,
      levels = c(
        "Non-responder",
        "Responder"
      )
    )
  )


# /////////////////////////////////////////////////////////////////////////////
# Overall cognitive-change summary
# /////////////////////////////////////////////////////////////////////////////

change_summary <- df_high_adh %>%
  summarise(
    n = n(),
    
    mean_change =
      mean(cog_change_2y, na.rm = TRUE),
    
    sd_change =
      sd(cog_change_2y, na.rm = TRUE),
    
    median_change =
      median(cog_change_2y, na.rm = TRUE),
    
    q1 =
      quantile(
        cog_change_2y,
        0.25,
        na.rm = TRUE
      ),
    
    q3 =
      quantile(
        cog_change_2y,
        0.75,
        na.rm = TRUE
      ),
    
    min =
      min(cog_change_2y, na.rm = TRUE),
    
    max =
      max(cog_change_2y, na.rm = TRUE)
  )

print(change_summary)


# /////////////////////////////////////////////////////////////////////////////
# Responder/non-responder distribution
# /////////////////////////////////////////////////////////////////////////////

responder_summary <- df_high_adh %>%
  count(
    responder,
    name = "n"
  ) %>%
  mutate(
    percent = 100 * n / sum(n)
  )

print(responder_summary)


# /////////////////////////////////////////////////////////////////////////////
# 10. Cognition according to responder status
# /////////////////////////////////////////////////////////////////////////////

cognition_by_responder <- df_high_adh %>%
  group_by(responder) %>%
  summarise(
    
    n = n(),
    
    baseline_mean =
      mean(
        ztotal_composite_BL2,
        na.rm = TRUE
      ),
    
    baseline_sd =
      sd(
        ztotal_composite_BL2,
        na.rm = TRUE
      ),
    
    year2_mean =
      mean(
        ztotal_composite_24kk,
        na.rm = TRUE
      ),
    
    year2_sd =
      sd(
        ztotal_composite_24kk,
        na.rm = TRUE
      ),
    
    change_mean =
      mean(
        cog_change_2y,
        na.rm = TRUE
      ),
    
    change_sd =
      sd(
        cog_change_2y,
        na.rm = TRUE
      ),
    
    change_median =
      median(
        cog_change_2y,
        na.rm = TRUE
      ),
    
    .groups = "drop"
  )

print(cognition_by_responder)


# /////////////////////////////////////////////////////////////////////////////
# 11. Histogram
# /////////////////////////////////////////////////////////////////////////////

p_hist <- ggplot(
  df_high_adh,
  aes(x = cog_change_2y)
) +
  
  geom_histogram(
    bins = 30,
    fill = "grey70",
    color = "white"
  ) +
  
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    linewidth = 0.8
  ) +
  
  labs(
    x = "2-year change in global cognitive composite",
    y = "Number of participants"
  ) +
  
  theme_classic(
    base_size = 14
  )

print(p_hist)


# /////////////////////////////////////////////////////////////////////////////
# Histogram according to responder status
# /////////////////////////////////////////////////////////////////////////////

p_hist_responder <- ggplot(
  df_high_adh,
  aes(
    x = cog_change_2y,
    fill = responder
  )
) +
  
  geom_histogram(
    bins = 30,
    color = "white"
  ) +
  
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    linewidth = 0.8
  ) +
  
  scale_fill_manual(
    values = c(
      "Non-responder" = "#4D4D4D",
      "Responder" = "#CC79A7"
    )
  ) +
  
  labs(
    x = "2-year change in global cognitive composite",
    y = "Number of participants",
    fill = NULL
  ) +
  
  theme_classic(
    base_size = 14
  ) +
  
  theme(
    legend.position = "top"
  )

print(p_hist_responder)


# /////////////////////////////////////////////////////////////////////////////
# 13. Density plot
# /////////////////////////////////////////////////////////////////////////////

p_density <- ggplot(
  df_high_adh,
  aes(x = cog_change_2y)
) +
  
  geom_density(
    fill = "grey75",
    alpha = 0.5,
    linewidth = 0.8
  ) +
  
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    linewidth = 0.8
  ) +
  
  labs(
    x = "2-year change in global cognitive composite",
    y = "Density"
  ) +
  
  theme_classic(
    base_size = 14
  )

print(p_density)


# /////////////////////////////////////////////////////////////////////////////
# Save summary tables
# /////////////////////////////////////////////////////////////////////////////

write_csv(
  change_summary,
  paste0(
    output_path,
    "high_adherence_cognitive_change_summary.csv"
  )
)

write_csv(
  responder_summary,
  paste0(
    output_path,
    "high_adherence_responder_distribution.csv"
  )
)

write_csv(
  cognition_by_responder,
  paste0(
    output_path,
    "high_adherence_cognition_by_responder.csv"
  )
)


# /////////////////////////////////////////////////////////////////////////////
# Save participant-level responder dataset
# /////////////////////////////////////////////////////////////////////////////

write_csv(
  df_high_adh,
  paste0(
    output_path,
    "high_adherence_responder_dataset.csv"
  )
)


# /////////////////////////////////////////////////////////////////////////////
# Save plots
# /////////////////////////////////////////////////////////////////////////////

ggsave(
  paste0(
    output_path,
    "high_adherence_cognitive_change_histogram.pdf"
  ),
  p_hist,
  width = 7,
  height = 5
)

ggsave(
  paste0(
    output_path,
    "high_adherence_cognitive_change_by_responder.pdf"
  ),
  p_hist_responder,
  width = 7,
  height = 5
)

ggsave(
  paste0(
    output_path,
    "high_adherence_cognitive_change_density.pdf"
  ),
  p_density,
  width = 7,
  height = 5
)


# /////////////////////////////////////////////////////////////////////////////
# Console summary
# /////////////////////////////////////////////////////////////////////////////

cat("\n")
cat("============================================================\n")
cat("HIGH-ADHERENCE COGNITIVE RESPONSE ANALYSIS\n")
cat("============================================================\n\n")

cat(
  "High-adherence participants with baseline and 2-year cognition:",
  nrow(df_high_adh),
  "\n\n"
)

print(responder_summary)

cat("\n")

print(cognition_by_responder)

cat("\n============================================================\n")

# /////////////////////////////////////////////////////////////////////////////
# Compare responders vs non-responders
# /////////////////////////////////////////////////////////////////////////////

# ---------------------------------------------------------------------------
# 1. Baseline cognition
# ---------------------------------------------------------------------------

# Descriptive statistics
df_high_adh %>%
  group_by(responder) %>%
  summarise(
    n = n(),
    mean_baseline = mean(ztotal_composite_BL2, na.rm = TRUE),
    sd_baseline = sd(ztotal_composite_BL2, na.rm = TRUE),
    median_baseline = median(ztotal_composite_BL2, na.rm = TRUE),
    q1 = quantile(ztotal_composite_BL2, 0.25, na.rm = TRUE),
    q3 = quantile(ztotal_composite_BL2, 0.75, na.rm = TRUE),
    .groups = "drop"
  )

# Welch independent-samples t-test
t.test(
  ztotal_composite_BL2 ~ responder,
  data = df_high_adh
)

ggplot(
  df_high_adh,
  aes(
    x = responder,
    y = ztotal_composite_BL2
  )
) +
  geom_boxplot(
    width = 0.55,
    outlier.shape = NA
  ) +
  geom_jitter(
    width = 0.15,
    alpha = 0.4,
    size = 1.5
  ) +
  labs(
    x = NULL,
    y = "Baseline global cognitive composite"
  ) +
  theme_classic(base_size = 14)


# /////////////////////////////////////////////////////////////////////////////
# Cognitive trajectories: responders vs non-responders
# /////////////////////////////////////////////////////////////////////////////

df_responder_long <- df_high_adh %>%
  select(
    ID,
    responder,
    ztotal_composite_BL2,
    ztotal_composite_12kk,
    ztotal_composite_24kk
  ) %>%
  pivot_longer(
    cols = c(
      ztotal_composite_BL2,
      ztotal_composite_12kk,
      ztotal_composite_24kk
    ),
    names_to = "timepoint",
    values_to = "cognition"
  ) %>%
  mutate(
    year = case_when(
      timepoint == "ztotal_composite_BL2"  ~ 0,
      timepoint == "ztotal_composite_12kk" ~ 1,
      timepoint == "ztotal_composite_24kk" ~ 2
    )
  )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

p_responder_trajectory <- ggplot(
  df_responder_long,
  aes(
    x = year,
    y = cognition,
    group = responder,
    color = responder
  )
) +
  stat_summary(
    fun = mean,
    geom = "line",
    linewidth = 1
  ) +
  stat_summary(
    fun = mean,
    geom = "point",
    size = 3
  ) +
  stat_summary(
    fun.data = mean_se,
    geom = "errorbar",
    width = 0.08
  ) +
  scale_x_continuous(
    breaks = c(0, 1, 2)
  ) +
  labs(
    x = "Years",
    y = "Global cognitive composite",
    color = NULL
  ) +
  theme_classic(base_size = 14)


# Display plot
p_responder_trajectory


# ---------------------------------------------------------------------------
# Save SVG
# ---------------------------------------------------------------------------

ggsave(
  filename = paste0(output_path, "cognitive_trajectories_responders_vs_nonresponders_high_adherence.svg"),
  plot = p_responder_trajectory,
  width = 8,
  height = 6.5,
  units = "in"
)


# ---------------------------------------------------------------------------
# Save PNG, 600 dpi
# ---------------------------------------------------------------------------

ggsave(
  filename = paste0(output_path, "cognitive_trajectories_responders_vs_nonresponders_high_adherence.png"),
  plot = p_responder_trajectory,
  width = 8,
  height = 6.5,
  units = "in",
  dpi = 600
)

trajectory_summary <- df_responder_long %>%
  group_by(
    responder,
    year
  ) %>%
  summarise(
    n = sum(!is.na(cognition)),
    mean = mean(cognition, na.rm = TRUE),
    sd = sd(cognition, na.rm = TRUE),
    se = sd / sqrt(n),
    .groups = "drop"
  )

write_csv(
  trajectory_summary,
  paste0(output_path, "cognitive_trajectories_responders_vs_nonresponders_high_adherence.csv")
)

trajectory_summary

# /////////////////////////////////////////////////////////////////////////////
# Mean 2-year cognitive change by study group
# /////////////////////////////////////////////////////////////////////////////


# Calculate individual 2-year change
df_change <- df_all %>%
  mutate(
    cognitive_change =
      ztotal_composite_24kk - ztotal_composite_BL2
  )


# Helper function
summarise_change <- function(data, group_name) {
  
  data %>%
    filter(
      !is.na(ztotal_composite_BL2),
      !is.na(ztotal_composite_24kk)
    ) %>%
    summarise(
      Group = group_name,
      N = n(),
      Baseline_mean = mean(ztotal_composite_BL2),
      Year2_mean = mean(ztotal_composite_24kk),
      Mean_change = mean(cognitive_change),
      SD_change = sd(cognitive_change),
      SE_change = SD_change / sqrt(N),
      CI_lower = Mean_change - 1.96 * SE_change,
      CI_upper = Mean_change + 1.96 * SE_change
    )
}


# ---------------------------------------------------------------------------
# Total population
# ---------------------------------------------------------------------------

change_total <- summarise_change(
  df_change,
  "Total population"
)


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------

change_control <- summarise_change(
  df_change %>%
    filter(sub_sample_id == 3),
  "Control"
)


# ---------------------------------------------------------------------------
# Intervention
# ---------------------------------------------------------------------------

change_intervention <- summarise_change(
  df_change %>%
    filter(sub_sample_id != 3),
  "Intervention"
)


# ---------------------------------------------------------------------------
# High-adherence intervention
# ---------------------------------------------------------------------------

change_high_adherence <- summarise_change(
  df_change %>%
    filter(adherence_class == "high"),
  "High adherence"
)


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------

cognitive_change_table <- bind_rows(
  change_total,
  change_control,
  change_intervention,
  change_high_adherence
)


# Round for viewing
cognitive_change_table_display <- cognitive_change_table %>%
  mutate(
    across(
      c(
        Baseline_mean,
        Year2_mean,
        Mean_change,
        SD_change,
        SE_change,
        CI_lower,
        CI_upper
      ),
      ~ round(.x, 3)
    )
  )


print(
  cognitive_change_table_display,
  n = Inf
)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

write_csv(
  cognitive_change_table,
  paste0(output_path, "mean_2year_cognitive_change_by_group.csv")
)

# /////////////////////////////////////////////////////////////////////////////
# Observed mean 2-year cognitive change
# /////////////////////////////////////////////////////////////////////////////


df_cog_change <- df_all %>%
  
  # Randomized groups only
  filter(
    sub_sample_id %in% c(2, 3)
  ) %>%
  
  # Baseline AND 2-year cognition required
  filter(
    !is.na(ztotal_composite_BL2),
    !is.na(ztotal_composite_24kk)
  ) %>%
  
  mutate(
    cognitive_change =
      ztotal_composite_24kk -
      ztotal_composite_BL2
  )


# ---------------------------------------------------------------------------
# Summary function
# ---------------------------------------------------------------------------

summarise_cognitive_change <- function(data, group_name) {
  
  data %>%
    summarise(
      
      Group = group_name,
      
      N = n(),
      
      Baseline_mean =
        mean(ztotal_composite_BL2),
      
      Baseline_SD =
        sd(ztotal_composite_BL2),
      
      Year2_mean =
        mean(ztotal_composite_24kk),
      
      Year2_SD =
        sd(ztotal_composite_24kk),
      
      Mean_change =
        mean(cognitive_change),
      
      SD_change =
        sd(cognitive_change),
      
      SE_change =
        SD_change / sqrt(N),
      
      CI_lower =
        Mean_change -
        qt(0.975, df = N - 1) * SE_change,
      
      CI_upper =
        Mean_change +
        qt(0.975, df = N - 1) * SE_change
    )
}


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

change_total <- summarise_cognitive_change(
  df_cog_change,
  "Total population"
)

change_control <- summarise_cognitive_change(
  df_cog_change %>%
    filter(sub_sample_id == 3),
  "Control"
)

change_intervention <- summarise_cognitive_change(
  df_cog_change %>%
    filter(sub_sample_id == 2),
  "Intervention"
)

change_high_adherence <- summarise_cognitive_change(
  df_cog_change %>%
    filter(
      sub_sample_id == 2,
      adherence_class == "high"
    ),
  "High adherence"
)


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------

cognitive_change_table <- bind_rows(
  change_total,
  change_control,
  change_intervention,
  change_high_adherence
)

print(
  cognitive_change_table,
  n = Inf
)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

write_csv(
  cognitive_change_table,
  paste0(output_path, "observed_mean_2year_cognitive_change_by_group.csv")
)
# =========================================================
# Logistic regression:
# Baseline NULISA biomarkers -> cognitive improvement
# High-adherence intervention participants only
#
# Outcome:
#   responder_binary = 1 if 2-year cognition - baseline cognition > 0
#   responder_binary = 0 if 2-year cognition - baseline cognition <= 0
#
# Biomarker preprocessing (NPQ version):
#   - NULISA NPQ used as is (already log2); no half-minimum replacement
#   - winsorize at 1st / 99th percentile
#   - z-standardize
#   - APOE4 and core AD biomarkers excluded
#
# Adjustment:
#   baseline cognition
#   age
#   sex
#   education
#   BMI
#   study site
#
# Reference categories:
#   sex = 1
#   campid = 101 (Helsinki)
#
# Output:
#   beta, SE, OR, 95% CI, raw P, BH-FDR
# =========================================================


# ---------------------------------------------------------
# 2. Identify NULISA biomarkers
# ---------------------------------------------------------

nulisa_0_cols <- names(df_all)[
  grepl(
    "^nulisa_.*_0$",
    names(df_all)
  )
]

nulisa_2_cols <- names(df_all)[
  grepl(
    "^nulisa_.*_2$",
    names(df_all)
  )
]


# Keep biomarkers that have both baseline and 2-year measurements
nulisa_bases <- str_remove(
  nulisa_0_cols,
  "_0$"
)

nulisa_bases <- nulisa_bases[
  paste0(
    nulisa_bases,
    "_2"
  ) %in% nulisa_2_cols
]


# NULISA only for primary responder analysis
biomarker_names <- nulisa_bases
biomarker_names <- drop_excluded(biomarker_names)


cat(
  "Number of NULISA biomarkers:",
  length(biomarker_names),
  "\n"
)


# ---------------------------------------------------------
# 3. Create responder analysis population
# ---------------------------------------------------------

df_resp <- df_all %>%
  
  filter(
    adherence_class == "high",
    !is.na(ztotal_composite_BL2),
    !is.na(ztotal_composite_24kk)
  ) %>%
  
  mutate(
    
    # Observed 2-year cognitive change
    cog_change_2y =
      ztotal_composite_24kk -
      ztotal_composite_BL2,
    
    # Binary outcome
    responder_binary =
      if_else(
        cog_change_2y > 0,
        1L,
        0L
      ),
    
    # Categorical variables
    sex = factor(sex),
    
    campid = factor(campid),
    
    # BMI preprocessing consistent with previous analyses
    bmi_bl2_z =
      as.numeric(
        scale(bmi_bl2)
      )
  )


# ---------------------------------------------------------
# 4. Reference categories
# ---------------------------------------------------------

if ("1" %in% levels(df_resp$sex)) {
  
  df_resp$sex <- relevel(
    df_resp$sex,
    ref = "1"
  )
}


if ("101" %in% levels(df_resp$campid)) {
  
  df_resp$campid <- relevel(
    df_resp$campid,
    ref = "101"
  )
}


# ---------------------------------------------------------
# 5. Check responder distribution
# ---------------------------------------------------------

cat(
  "\nResponder distribution:\n"
)

print(
  table(
    df_resp$responder_binary
  )
)

print(
  round(
    prop.table(
      table(
        df_resp$responder_binary
      )
    ) * 100,
    1
  )
)



# ---------------------------------------------------------
# 6. One-biomarker logistic regression
# ---------------------------------------------------------

run_responder_biomarker <- function(
    bm_base,
    data
) {
  
  cat(
    "Running:",
    bm_base,
    "\n"
  )
  
  
  # -------------------------------------------------------
  # Select correct baseline biomarker column
  # -------------------------------------------------------
  
  # exactly as in previous NULISA analyses
  bm0 <- paste0(
      bm_base,
      "_0"
    )
  
  
  # -------------------------------------------------------
  # Check column exists
  # -------------------------------------------------------
  
  if (!(bm0 %in% names(data))) {
    
    return(
      tibble(
        biomarker =
          str_remove(
            bm_base,
            "^nulisa_"
          ),
        
        n = NA_integer_,
        n_responder = NA_integer_,
        n_nonresponder = NA_integer_,
        
        beta = NA_real_,
        SE = NA_real_,
        
        OR = NA_real_,
        CI_low = NA_real_,
        CI_high = NA_real_,
        
        p_value = NA_real_,
        
        error =
          paste(
            "Missing column:",
            bm0
          )
      )
    )
  }
  
  
  # -------------------------------------------------------
  # Complete-case analytic sample
  # -------------------------------------------------------
  
  df_model <- data %>%
    
    select(
      ID,
      responder_binary,
      all_of(bm0),
      
      ztotal_composite_BL2,
      
      age_bl1,
      sex,
      educ_years_BL1,
      bmi_bl2_z,
      campid
    ) %>%
    
    filter(
      !is.na(responder_binary),
      !is.na(.data[[bm0]]),
      
      !is.na(ztotal_composite_BL2),
      
      !is.na(age_bl1),
      !is.na(sex),
      !is.na(educ_years_BL1),
      !is.na(bmi_bl2_z),
      !is.na(campid)
    )
  
  
  # -------------------------------------------------------
  # Winsorization
  # -------------------------------------------------------
  
  df_model <- df_model %>%
    
    mutate(
      bm0_w =
        winsorize(
          .data[[bm0]],
          p = 0.01
        )
    )
  
  
  # -------------------------------------------------------
  # Check positivity before log
  # -------------------------------------------------------
  
  if (
    (needs_log(bm_base) && any(df_model$bm0_w <= 0, na.rm = TRUE))
  ) {
    
    return(
      tibble(
        biomarker =
          str_remove(
            bm_base,
            "^nulisa_"
          ),
        
        n = nrow(df_model),
        
        n_responder =
          sum(
            df_model$responder_binary == 1
          ),
        
        n_nonresponder =
          sum(
            df_model$responder_binary == 0
          ),
        
        beta = NA_real_,
        SE = NA_real_,
        
        OR = NA_real_,
        CI_low = NA_real_,
        CI_high = NA_real_,
        
        p_value = NA_real_,
        
        error =
          "Non-positive values after winsorization; log not possible"
      )
    )
  }
  
  
  # -------------------------------------------------------
  # Log transformation
  # -------------------------------------------------------
  
  df_model <- df_model %>%
    
    mutate(
      bm0_log =
        log_transformation(bm0_w, bm_base)
    )
  
  
  # -------------------------------------------------------
  # Z-standardization
  # -------------------------------------------------------
  
  df_model <- df_model %>%
    
    mutate(
      bm0_log_z =
        zscore_safe(
          bm0_log
        )
    ) %>%
    
    filter(
      !is.na(bm0_log_z)
    )
  
  
  # -------------------------------------------------------
  # Check outcome after complete-case filtering
  # -------------------------------------------------------
  
  n_resp <- sum(
    df_model$responder_binary == 1
  )
  
  n_nonresp <- sum(
    df_model$responder_binary == 0
  )
  
  
  if (
    n_resp == 0 |
    n_nonresp == 0
  ) {
    
    return(
      tibble(
        biomarker =
          str_remove(
            bm_base,
            "^nulisa_"
          ),
        
        n = nrow(df_model),
        
        n_responder = n_resp,
        n_nonresponder = n_nonresp,
        
        beta = NA_real_,
        SE = NA_real_,
        
        OR = NA_real_,
        CI_low = NA_real_,
        CI_high = NA_real_,
        
        p_value = NA_real_,
        
        error =
          "Only one responder category available"
      )
    )
  }
  
  
  # -------------------------------------------------------
  # Logistic regression
  # -------------------------------------------------------
  
  model <- tryCatch(
    
    glm(
      responder_binary ~
        
        bm0_log_z +
        
        # Baseline cognition
        ztotal_composite_BL2 +
        
        # Standard FINGER covariates
        age_bl1 +
        sex +
        educ_years_BL1 +
        bmi_bl2_z +
        campid,
      
      family =
        binomial(
          link = "logit"
        ),
      
      data = df_model
    ),
    
    error = function(e) e
  )
  
  
  # -------------------------------------------------------
  # Catch model errors
  # -------------------------------------------------------
  
  if (
    inherits(
      model,
      "error"
    )
  ) {
    
    return(
      tibble(
        biomarker =
          str_remove(
            bm_base,
            "^nulisa_"
          ),
        
        n = nrow(df_model),
        
        n_responder = n_resp,
        n_nonresponder = n_nonresp,
        
        beta = NA_real_,
        SE = NA_real_,
        
        OR = NA_real_,
        CI_low = NA_real_,
        CI_high = NA_real_,
        
        p_value = NA_real_,
        
        error =
          model$message
      )
    )
  }
  
  
  # -------------------------------------------------------
  # Extract protein coefficient
  # -------------------------------------------------------
  
  fixed <- broom::tidy(
    model
  )
  
  
  bm_result <- fixed %>%
    
    filter(
      term == "bm0_log_z"
    )
  
  
  if (
    nrow(
      bm_result
    ) != 1
  ) {
    
    return(
      tibble(
        biomarker =
          str_remove(
            bm_base,
            "^nulisa_"
          ),
        
        n = nrow(df_model),
        
        n_responder = n_resp,
        n_nonresponder = n_nonresp,
        
        beta = NA_real_,
        SE = NA_real_,
        
        OR = NA_real_,
        CI_low = NA_real_,
        CI_high = NA_real_,
        
        p_value = NA_real_,
        
        error =
          "Biomarker coefficient not found"
      )
    )
  }
  
  
  # -------------------------------------------------------
  # Results
  # -------------------------------------------------------
  
  beta <- bm_result$estimate
  
  SE <- bm_result$std.error
  
  
  tibble(
    
    biomarker =
      str_remove(
        bm_base,
        "^nulisa_"
      ),
    
    n =
      nrow(
        df_model
      ),
    
    n_responder =
      n_resp,
    
    n_nonresponder =
      n_nonresp,
    
    beta =
      beta,
    
    SE =
      SE,
    
    OR =
      exp(
        beta
      ),
    
    CI_low =
      exp(
        beta -
          1.96 * SE
      ),
    
    CI_high =
      exp(
        beta +
          1.96 * SE
      ),
    
    p_value =
      bm_result$p.value,
    
    error =
      NA_character_
  )
}


# ---------------------------------------------------------
# 7. Run all NULISA biomarkers
# ---------------------------------------------------------

results_responder <- map_dfr(
  
  biomarker_names,
  
  run_responder_biomarker,
  
  data = df_resp
)


# ---------------------------------------------------------
# 8. BH-FDR
# ---------------------------------------------------------

results_responder <- results_responder %>%
  
  mutate(
    
    FDR =
      p.adjust(
        p_value,
        method = "fdr"
      )
  ) %>%
  
  arrange(
    p_value
  )


# ---------------------------------------------------------
# 9. Add direction / significance labels
# ---------------------------------------------------------

results_responder <- results_responder %>%
  
  mutate(
    
    direction =
      case_when(
        
        beta > 0 ~
          "Higher odds of cognitive improvement",
        
        beta < 0 ~
          "Lower odds of cognitive improvement",
        
        TRUE ~
          NA_character_
      ),
    
    significance =
      case_when(
        
        FDR < 0.05 ~
          "FDR < 0.05",
        
        p_value < 0.05 ~
          "Nominal P < 0.05",
        
        TRUE ~
          "Not significant"
      )
  )


# ---------------------------------------------------------
# 10. Round output
# ---------------------------------------------------------

results_responder <- results_responder %>%
  
  mutate(
    across(
      where(is.numeric),
      ~ round(.x, 6)
    )
  )


# ---------------------------------------------------------
# 11. View results
# ---------------------------------------------------------

print(
  results_responder,
  n = Inf
)


cat(
  "\nFDR-significant biomarkers:\n"
)

print(
  results_responder %>%
    filter(
      FDR < 0.05
    ),
  n = Inf
)


cat(
  "\nNominally significant biomarkers:\n"
)

print(
  results_responder %>%
    filter(
      p_value < 0.05
    ),
  n = Inf
)


# ---------------------------------------------------------
# 12. Save
# ---------------------------------------------------------

write_csv(
  results_responder,
  paste0(
    output_path,
    "baseline_NULISA_responder_logistic_regression.csv"
  )
)

# =========================================================
# FIGURE 6: SUPPLEMENTARY TABLES
#
# Baseline characteristics and 2-year cognitive change
# Responders vs non-responders
#
# Population:
# EXACT common sample used in all logistic regressions
#
# Expected:
# Total N = 363
# Responders = 283
# Non-responders = 80
#
# Outputs:
# 1. baseline_characteristics_363.csv
# 2. cognitive_change_responders_363.csv
# =========================================================




# ---------------------------------------------------------
# 1. Identify exact common logistic regression population
# ---------------------------------------------------------

# Confirm all protein models have the expected sample size

# (previously fixed to 124 proteins / N = 363 / 283 / 80; now checked
#  dynamically because excluded proteins change the protein count)
ok_models <- results_responder %>% filter(is.na(error))
stopifnot(
  nrow(results_responder) == length(biomarker_names),
  n_distinct(ok_models$n) == 1,
  n_distinct(ok_models$n_responder) == 1
)
expected_n      <- ok_models$n[1]
expected_resp   <- ok_models$n_responder[1]
expected_nonres <- ok_models$n_nonresponder[1]
cat("\nProteins modelled:", nrow(results_responder),
    "| N =", expected_n, "| responders =", expected_resp,
    "| non-responders =", expected_nonres, "\n")


# Required covariates

required_vars <- c(
  "ztotal_composite_BL2",
  "age_bl1",
  "sex",
  "educ_years_BL1",
  "bmi_bl2_z",
  "campid"
)


# Start with participants meeting all non-protein criteria

df_common <- df_resp %>%
  
  filter(
    if_all(
      all_of(required_vars),
      ~ !is.na(.x)
    ),
    !is.na(responder_binary)
  )


# ---------------------------------------------------------
# 2. Apply baseline protein availability criteria
#
# Reconstruct the inclusion criteria from your
# logistic regression code for every protein.
# ---------------------------------------------------------

for (bm_base in biomarker_names) {
  
  bm0 <- paste0(
      bm_base,
      "_0"
    )

  
  if (!(bm0 %in% names(df_common))) {
    
    stop(
      paste(
        "Missing protein column:",
        bm0
      )
    )
  }
  
  
  # Restrict to participants with available protein
  
  df_common <- df_common %>%
    
    filter(
      !is.na(.data[[bm0]])
    )
  
}


# ---------------------------------------------------------
# 3. Verify population
# ---------------------------------------------------------

cat("\nCommon logistic regression population:\n")

print(
  df_common %>%
    count(responder_binary)
)

cat(
  "\nTotal N:",
  nrow(df_common),
  "\n"
)


stopifnot(
  nrow(df_common) == expected_n,
  sum(df_common$responder_binary == 1) == expected_resp,
  sum(df_common$responder_binary == 0) == expected_nonres,
  n_distinct(df_common$ID) == expected_n
)


# ---------------------------------------------------------
# 4. Prepare variables for supplementary tables
# ---------------------------------------------------------

df_table <- df_common %>%
  
  mutate(
    
    responder_group = factor(
      responder_binary,
      levels = c(0, 1),
      labels = c(
        "Non-responder",
        "Responder"
      )
    ),
    
    sex_label = factor(
      as.character(sex),
      levels = c("1", "2"),
      labels = c("Male", "Female")
    ),
    
    apoe4_label = factor(
      apoe4_status,
      levels = c(0, 1),
      labels = c(
        "Non-carrier",
        "Carrier"
      )
    ),
    
    site = factor(campid)
    
  )


# =========================================================
# TABLE 1: BASELINE CHARACTERISTICS
# =========================================================


baseline_table <- df_table %>%
  
  select(
    responder_group,
    age_bl1,
    sex_label,
    educ_years_BL1,
    bmi_bl2,
    apoe4_label,
    ztotal_composite_BL2,
    site
  ) %>%
  
  tbl_summary(
    
    by = responder_group,
    
    statistic = list(
      
      all_continuous() ~ "{mean} ({sd})",
      
      all_categorical() ~ "{n} ({p}%)"
      
    ),
    
    digits = all_continuous() ~ 2,
    
    missing = "ifany",
    
    label = list(
      
      age_bl1 ~ "Age, years",
      
      sex_label ~ "Sex",
      
      educ_years_BL1 ~ "Education, years",
      
      bmi_bl2 ~ "BMI, kg/m²",
      
      apoe4_label ~ "APOE ε4 carrier status",
      
      ztotal_composite_BL2 ~
        "Baseline global cognition",
      
      site ~ "Study site"
      
    )
    
  ) %>%
  
  add_p(
    
    test = list(
      
      all_continuous() ~ "wilcox.test",
      
      sex_label ~ "fisher.test",
      
      apoe4_label ~ "fisher.test",
      
      site ~ "fisher.test"
      
    )
    
  ) %>%
  
  add_overall() %>%
  
  bold_labels()


cat("\nBASELINE CHARACTERISTICS:\n")

print(baseline_table)


# ---------------------------------------------------------
# Save baseline characteristics
# ---------------------------------------------------------

write_csv(
  
  baseline_table %>%
    as_tibble(col_labels = FALSE),
  
  paste0(
    output_path,
    "baseline_characteristics_363.csv"
  )
  
)


# =========================================================
# TABLE 2: OBSERVED TWO-YEAR COGNITIVE CHANGE
# =========================================================


cognitive_change_table <- df_table %>%
  
  group_by(responder_group) %>%
  
  summarise(
    
    n = n(),
    
    baseline_mean =
      mean(
        ztotal_composite_BL2,
        na.rm = TRUE
      ),
    
    baseline_sd =
      sd(
        ztotal_composite_BL2,
        na.rm = TRUE
      ),
    
    year2_mean =
      mean(
        ztotal_composite_24kk,
        na.rm = TRUE
      ),
    
    year2_sd =
      sd(
        ztotal_composite_24kk,
        na.rm = TRUE
      ),
    
    mean_change =
      mean(
        cog_change_2y,
        na.rm = TRUE
      ),
    
    sd_change =
      sd(
        cog_change_2y,
        na.rm = TRUE
      ),
    
    median_change =
      median(
        cog_change_2y,
        na.rm = TRUE
      ),
    
    q1_change =
      quantile(
        cog_change_2y,
        0.25,
        na.rm = TRUE
      ),
    
    q3_change =
      quantile(
        cog_change_2y,
        0.75,
        na.rm = TRUE
      ),
    
    .groups = "drop"
    
  )


cat("\nTWO-YEAR COGNITIVE CHANGE:\n")

print(
  cognitive_change_table,
  width = Inf
)


# ---------------------------------------------------------
# Save cognitive change
# ---------------------------------------------------------

write_csv(
  
  cognitive_change_table,
  
  paste0(
    output_path,
    "cognitive_change_responders_363.csv"
  )
  
)


# =========================================================
# FINAL CHECKS
# =========================================================


cat("\nFINAL SAMPLE:\n")

cat(
  "Total:",
  nrow(df_table),
  "\n"
)

cat(
  "Responders:",
  sum(df_table$responder_binary == 1),
  "\n"
)

cat(
  "Non-responders:",
  sum(df_table$responder_binary == 0),
  "\n"
)


cat("\nFiles saved successfully.\n")

# =========================================================
# Volcano plot:
# Baseline NULISA biomarkers associated with
# cognitive improvement among high-adherence participants
#
# Uses: results_responder
#
# x-axis = logistic regression beta
# y-axis = -log10(raw p-value)
# Labels = top 15 proteins by raw p-value
#
# Positive beta:
# higher baseline protein -> higher odds of cognitive improvement
#
# Negative beta:
# higher baseline protein -> lower odds of cognitive improvement
# =========================================================



# ---------------------------------------------------------
# 1. Proteins to exclude
# ---------------------------------------------------------

exclude_proteins <- c(
  "pTau181",
  "pTau231",
  "pTau217",
  "AB38",
  "NEFL",
  "GFAP",
  "AB40",
  "AB42",
  "lumipulse_ptau217"
)


# ---------------------------------------------------------
# 2. Prepare volcano data
# ---------------------------------------------------------

volcano_resp_df <- results_responder %>%
  
  filter(
    !biomarker %in% exclude_proteins,
    !is.na(beta),
    !is.na(p_value),
    p_value > 0
  ) %>%
  
  mutate(
    
    neglog10p = -log10(p_value),
    
    status = case_when(
      
      p_value < 0.05 & beta > 0 ~
        "Associated with response, P < 0.05",
      
      p_value < 0.05 & beta < 0 ~
        "Associated with non-response, P < 0.05",
      
      p_value < 0.10 & beta > 0 ~
        "Associated with response, P < 0.10",
      
      p_value < 0.10 & beta < 0 ~
        "Associated with non-response, P < 0.10",
      
      TRUE ~
        "Not significant"
    )
  )


# ---------------------------------------------------------
# 3. Label all nominally significant proteins
# ---------------------------------------------------------

volcano_labels <- volcano_resp_df %>%
  filter(p_value < 0.05)


# ---------------------------------------------------------
# 4. Colors
# ---------------------------------------------------------

volcano_resp_colors <- c(
  "Associated with non-response, P < 0.05" =
    "#CC6677",
  "Associated with non-response, P < 0.10" =
    "#E7A1B0",
  "Not significant" =
    "grey70",
  "Associated with response, P < 0.10" =
    "#88CCAA",
  
  "Associated with response, P < 0.05" =
    "#228833"
)


# ---------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------

p_volcano_resp <- ggplot(
  
  volcano_resp_df,
  
  aes(
    x = beta,
    y = neglog10p,
    color = status
  )
  
) +
  
  # Raw P = 0.05
  geom_hline(
    yintercept = -log10(0.05),
    linetype = "dashed",
    linewidth = 0.5,
    color = "grey45"
  ) +
  
  # No association
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    linewidth = 0.5,
    color = "grey55"
  ) +
  
  geom_point(
    size = 2.8,
    alpha = 0.9
  ) +
  
  geom_text_repel(
    data = volcano_labels,
    aes(label = biomarker),
    size = 4,
    max.overlaps = Inf,
    box.padding = 0.4,
    point.padding = 0.3,
    segment.color = "grey50",
    show.legend = FALSE
  ) +
  
  scale_color_manual(
    values = volcano_resp_colors,
    breaks = c(
      "Associated with response, P < 0.05",
      "Associated with response, P < 0.10",
      "Associated with non-response, P < 0.05",
      "Associated with non-response, P < 0.10",
      "Not significant"
    )
  ) +
  
  labs(
    
    title =
      "Baseline proteins associated with cognitive response",
    
    subtitle =
      "High-adherence intervention participants",
    
    x =
      "Association with responder status (β)\n← Non-response                         Response →",
    
    y =
      expression(
        -log[10] * "(P value)"
      ),
    
    color = NULL
  ) +
  
  theme_classic(
    base_size = 12
  ) +
  
  theme(
    
    plot.title =
      element_text(
        face = "bold",
        size = 12
      ),
    
    plot.subtitle =
      element_text(
        size = 11
      ),
    
    legend.position =
      "right",
    
    legend.text =
      element_text(
        size = 9
      )
  )


print(p_volcano_resp)


# ---------------------------------------------------------
# 6. Save
# ---------------------------------------------------------
ggsave(
  filename = paste0(output_path, "volcano_baseline_NULISA_cognitive_improvement_high_adherencev2.svg"),
  plot = p_volcano_resp,
  width = 9,
  height = 7,
  units = "in"
)
ggsave(
  
  filename = paste0(output_path, "volcano_baseline_NULISA_cognitive_improvement_high_adherence.png"),
  
  plot =
    p_volcano_resp,
  
  width = 9,
  height = 7,
  dpi = 600
)


write_csv(
  volcano_resp_df,
  paste0(output_path, "volcano_baseline_NULISA_cognitive_improvement_high_adherence.csv")
)

# =========================================================
# Forest plot:
# Baseline NULISA biomarkers associated with
# cognitive improvement among high-adherence participants
#
# Uses: results_responder
#
# Shows:
# - proteins with raw p < 0.05
# - adjusted odds ratios
# - 95% confidence intervals
#
# OR > 1:
# higher baseline protein -> higher odds of cognitive improvement
#
# OR < 1:
# higher baseline protein -> lower odds of cognitive improvement
# =========================================================



# ---------------------------------------------------------
# 1. Proteins to exclude
# ---------------------------------------------------------

exclude_proteins <- c(
  "pTau181",
  "pTau231",
  "pTau217",
  "AB38",
  "NEFL",
  "GFAP",
  "AB40",
  "AB42",
  "lumipulse_ptau217"
)


# ---------------------------------------------------------
# 2. Prepare forest plot data
# ---------------------------------------------------------

forest_resp_df <- results_responder %>%
  
  filter(
    !biomarker %in% exclude_proteins,
    !is.na(OR),
    !is.na(CI_low),
    !is.na(CI_high),
    !is.na(p_value),
    p_value < 0.05
  ) %>%
  
  mutate(
    
    direction = case_when(
      
      OR > 1 ~
        "Higher odds of cognitive improvement",
      
      OR < 1 ~
        "Lower odds of cognitive improvement",
      
      TRUE ~
        "No association"
    ),
    
    # Text for optional table/output
    OR_CI = sprintf(
      "%.2f (%.2f–%.2f)",
      OR,
      CI_low,
      CI_high
    ),
    
    p_label = case_when(
      p_value < 0.001 ~ "<0.001",
      TRUE ~ sprintf("%.3f", p_value)
    )
  ) %>%
  
  # Order according to effect size
  arrange(OR) %>%
  
  mutate(
    biomarker = factor(
      biomarker,
      levels = biomarker
    )
  )


# ---------------------------------------------------------
# 3. Check data
# ---------------------------------------------------------

print(
  forest_resp_df %>%
    select(
      biomarker,
      OR,
      CI_low,
      CI_high,
      p_value,
      FDR,
      direction
    ),
  n = Inf
)


# ---------------------------------------------------------
# 4. Colors
# ---------------------------------------------------------

forest_colors <- c(
  
  "Lower odds of cognitive improvement" =
    "#CC6677",
  
  "Higher odds of cognitive improvement" =
    "#228833"
)


# ---------------------------------------------------------
# 5. Forest plot
# ---------------------------------------------------------

forest_n <- results_responder %>% filter(is.na(error))

p_forest_resp <- ggplot(
  
  forest_resp_df,
  
  aes(
    x = OR,
    y = biomarker,
    color = direction
  )
  
) +
  
  # Null association
  geom_vline(
    xintercept = 1,
    linetype = "dashed",
    linewidth = 0.6,
    color = "grey50"
  ) +
  
  # 95% CI
  geom_errorbarh(
    aes(
      xmin = CI_low,
      xmax = CI_high
    ),
    height = 0.18,
    linewidth = 0.7
  ) +
  
  # OR
  geom_point(
    size = 3
  ) +
  
  scale_color_manual(
    values = forest_colors
  ) +
  
  labs(
    
    title =
      "Baseline proteins associated with cognitive improvement",
    
    subtitle =
      sprintf("High-adherence intervention participants (n=%d; %d responders, %d non-responders)",
              forest_n$n[1], forest_n$n_responder[1], forest_n$n_nonresponder[1]),
    
    x =
      "Adjusted odds ratio per 1-SD higher baseline protein",
    
    y = NULL,
    
    color = NULL
  ) +
  
  theme_classic(
    base_size = 12
  ) +
  
  theme(
    
    plot.title =
      element_text(
        face = "bold",
        size = 12
      ),
    
    plot.subtitle =
      element_text(
        size = 10
      ),
    
    axis.text.y =
      element_text(
        size = 10
      ),
    
    legend.position =
      "bottom",
    
    legend.text =
      element_text(
        size = 9
      )
  )


print(
  p_forest_resp
)


# ---------------------------------------------------------
# 6. Save plot
# ---------------------------------------------------------
# SVG
ggsave(
  filename = paste0(output_path, "forest_baseline_NULISA_cognitive_improvement_high_adherence.svg"),
  plot = p_forest_resp,
  width = 8,
  height = 6.5,
  units = "in"
)

# PNG, 600 dpi
ggsave(
  filename = paste0(output_path, "forest_baseline_NULISA_cognitive_improvement_high_adherence.png"),
  plot = p_forest_resp,
  width = 8,
  height = 6.5,
  dpi = 600
)

# Data
write_csv(
  forest_resp_df,
  paste0(output_path, "forest_baseline_NULISA_cognitive_improvement_high_adherence.csv")
)
