
# ============================================================
# FINGER NULISA PROTEOMICS
# SUPPLEMENTARY BASELINE CORRELATION MATRICES
#
# MATRIX A:
# Exploratory NULISA proteins vs:
# Age, Sex, APOE epsilon4 status
#
# MATRIX B:
# Exploratory NULISA proteins vs:
# pTau181, pTau231, pTau217, AB42/40,
# NEFL, GFAP, Lumipulse p-tau217
#
# All NULISA baseline columns end in _0.
#
# NPQ VERSION:
# - NULISA values are NPQ (already log2). No transformation is applied
#   (Spearman correlations are rank-based, so a monotone transform would
#   not change rho anyway).
# - Original columns are used (no half-minimum zero replacement).
# - NULISA APOE4 is excluded from the protein rows (genotype-driven).
#   APOE e4 carrier STATUS (genotype) is kept as a demographic variable.
# - AB42/40 on the log2 (NPQ) scale = AB42 - AB40, not AB42 / AB40.
#
# Sex recoding:
# Original: 1 = male, 2 = female
# Analysis: 0 = male, 1 = female
#
# Method: Spearman rank correlation
#
# Outputs:
# - Editable SVG figures
# - TIFF figures (600 dpi)
# - CSV files with rho, P, FDR, and N
# ============================================================


# ------------------------------------------------------------
# 1. PACKAGES
# ------------------------------------------------------------

packages <- c(
  "readr",
  "dplyr",
  "tidyr",
  "purrr",
  "ggplot2",
  "svglite",
  "patchwork"
)

new_packages <- setdiff(
  packages,
  rownames(installed.packages())
)

if (length(new_packages) > 0) {
  install.packages(new_packages)
}


# Loop to load packages
for (pkg in packages) {
  library(pkg, character.only = TRUE)
}

# ============================================================
# LOAD DATA AND APPLY PARTICIPANT SELECTION CRITERIA
# ============================================================
folder_path <- "../data/"
processed_path <- paste0(folder_path,"processed/")
output_path <- paste0(folder_path,"outputs_R/correlations/")


if (!dir.exists(output_path)) {dir.create(output_path, recursive = TRUE)}# Create the folder if it does not exist


# .............................................................................
# Load data
# .............................................................................
file_path <- paste0(processed_path,"dataset_original_clean.csv")
df <- read_csv(file_path, show_col_types = FALSE)

cat("Original participants/rows:", nrow(df), "\n")


# ------------------------------------------------------------
# 1. Required cognitive variables
# ------------------------------------------------------------

baseline_cognition <- "ztotal_composite_BL2"

followup_cognition <- c(
  "ztotal_composite_12kk",
  "ztotal_composite_24kk"
)


# ------------------------------------------------------------
# 2. Required covariates
# ------------------------------------------------------------

covariates <- c(
  "age_bl1",
  "sex",
  "educ_years_BL1",
  "bmi_bl2",
  "campid"
)


# ------------------------------------------------------------
# 3. Check variables exist
# ------------------------------------------------------------

required_vars <- c(
  "ID",
  baseline_cognition,
  followup_cognition,
  covariates
)

missing_vars <- setdiff(required_vars, names(df))

if (length(missing_vars) > 0) {
  stop(
    "Missing required variables:\n",
    paste(missing_vars, collapse = "\n")
  )
}


# ------------------------------------------------------------
# 4. Apply participant-selection criteria
# ------------------------------------------------------------

df <- df %>%
  dplyr::filter(
    # Baseline cognition must be available
    !is.na(ztotal_composite_BL2),
    
    # At least one follow-up must be available
    !is.na(ztotal_composite_12kk) |
      !is.na(ztotal_composite_24kk),
    
    # All specified covariates must be available
    dplyr::if_all(
      dplyr::all_of(covariates),
      ~ !is.na(.x)
    )
  )

cat(
  "Participants/rows after cognitive and covariate filtering:",
  nrow(df),
  "\n"
)


# ------------------------------------------------------------
# 5. Verify one row per participant
# ------------------------------------------------------------

if (anyDuplicated(df$ID) > 0) {
  stop(
    "Duplicate participant IDs detected. ",
    "Check whether the dataset contains repeated visits."
  )
}


# ------------------------------------------------------------
# 6. Check APOE4 status
# ------------------------------------------------------------

# APOE4 status is required for the demographic correlation,
# but was not included in your base_vars vector.
#
# Therefore, do not exclude participants missing APOE4
# from every correlation unless that was also done
# in the original longitudinal analysis.

if (!"apoe4_status" %in% names(df)) {
  stop("Column apoe4_status not found.")
}

cat(
  "Participants with available APOE4 status:",
  sum(!is.na(df$apoe4_status)),
  "\n"
)

# ------------------------------------------------------------
# 3. IDENTIFY ALL BASELINE NULISA PROTEINS
# ------------------------------------------------------------

# Example:
# nulisa_RUVBL2_0
# nulisa_TREM2_0
# nulisa_AB42_0

all_baseline_proteins <- grep(
  "^nulisa_.*_0$",
  names(df),
  value = TRUE
)

if (length(all_baseline_proteins) == 0) {
  stop(
    "No baseline NULISA columns found. ",
    "Expected names such as nulisa_RUVBL2_0."
  )
}


# ------------------------------------------------------------
# 4. EXCLUDE CORE BIOMARKERS FROM PROTEIN ROWS AND APOE4
# ------------------------------------------------------------

exclude_proteins <- c(
  "pTau181",
  "pTau231",
  "pTau217",
  "AB38",
  "NEFL",
  "GFAP",
  "AB40",
  "AB42",
  "lumipulse_ptau217",
  "APOE4"
)

# Remove prefix and baseline suffix for comparison

protein_names_clean <- sub(
  "_0$",
  "",
  sub("^nulisa_", "", all_baseline_proteins)
)

nulisa_proteins <- all_baseline_proteins[
  !protein_names_clean %in% exclude_proteins
]

# Alphabetical ordering, identical across matrices

nulisa_proteins <- nulisa_proteins[
  order(
    sub("_0$", "", sub("^nulisa_", "", nulisa_proteins))
  )
]

cat(
  "Exploratory proteins included as rows:",
  length(nulisa_proteins),
  "\n"
)


# ------------------------------------------------------------
# 5. CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns <- c(
  "age_bl1",
  "sex",
  "apoe4_status",
  "nulisa_AB42_0",
  "nulisa_AB40_0"
)

missing_columns <- setdiff(
  required_columns,
  names(df)
)

if (length(missing_columns) > 0) {
  stop(
    "Missing required columns:\n",
    paste(missing_columns, collapse = "\n")
  )
}


# ------------------------------------------------------------
# 6. RECODE SEX
# ------------------------------------------------------------

# Original:
# 1 = male
# 2 = female
#
# New:
# 0 = male
# 1 = female

if (!all(na.omit(df$sex) %in% c(1, 2))) {
  stop(
    "Unexpected sex coding. Expected 1 = male, 2 = female."
  )
}

if (!all(na.omit(df$apoe4_status) %in% c(0, 1))) {
  stop(
    "Unexpected APOE4 coding. Expected 0/1."
  )
}

df <- df %>%
  mutate(
    sex_female = case_when(
      sex == 1 ~ 0,
      sex == 2 ~ 1,
      TRUE ~ NA_real_
    )
  )


# ------------------------------------------------------------
# 7. CALCULATE BASELINE AB42/40 RATIO (log2 scale)
# ------------------------------------------------------------

# NPQ values are log2, so the log2 AB42/40 ratio is the DIFFERENCE:
#   log2(AB42/AB40) = NPQ_AB42 - NPQ_AB40
# (dividing two log2 values would not give the ratio)

df <- df %>%
  mutate(
    AB42_40_ratio = nulisa_AB42_0 - nulisa_AB40_0
  )

cat(
  "Nonmissing AB42/40 ratios:",
  sum(!is.na(df$AB42_40_ratio)),
  "\n"
)


# ------------------------------------------------------------
# 8. DEFINE CORRELATION VARIABLES
# ------------------------------------------------------------

demographic_variables <- c(
  "age_bl1",
  "sex_female",
  "apoe4_status"
)

# Map biomarker labels to baseline dataset columns.
#
# This assumes the six biomarkers use baseline _0 names.
# The script checks these names before running.

biomarker_columns <- c(
  "nulisa_pTau181_0",
  "nulisa_pTau231_0",
  "nulisa_pTau217_0",
  "AB42_40_ratio",
  "nulisa_NEFL_0",
  "nulisa_GFAP_0",
  "lumipulse_ptau217_0"
)


# ------------------------------------------------------------
# 9. CHECK ALL ANALYSIS COLUMNS
# ------------------------------------------------------------

required_analysis_columns <- unique(c(
  nulisa_proteins,
  demographic_variables,
  biomarker_columns
))

missing_analysis_columns <- setdiff(
  required_analysis_columns,
  names(df)
)

if (length(missing_analysis_columns) > 0) {
  stop(
    "These analysis columns are missing:\n",
    paste(missing_analysis_columns, collapse = "\n"),
    "\n\nCheck whether these biomarkers have a different ",
    "baseline column name."
  )
}


# ------------------------------------------------------------
# 10. ENSURE NUMERIC VALUES
# ------------------------------------------------------------

analysis_columns <- unique(c(
  nulisa_proteins,
  demographic_variables,
  biomarker_columns
))

non_numeric <- analysis_columns[
  !vapply(
    df[analysis_columns],
    is.numeric,
    logical(1)
  )
]

if (length(non_numeric) > 0) {
  stop(
    "These columns are not numeric:\n",
    paste(non_numeric, collapse = "\n")
  )
}


# ------------------------------------------------------------
# 11. SPEARMAN CORRELATION FUNCTION
# ------------------------------------------------------------

calculate_correlations <- function(
    data,
    proteins,
    variables
) {
  
  tidyr::expand_grid(
    protein = proteins,
    variable = variables
  ) %>%
    
    mutate(
      result = purrr::map2(
        protein,
        variable,
        
        function(protein, variable) {
          
          x <- data[[protein]]
          y <- data[[variable]]
          
          valid <- is.finite(x) & is.finite(y)
          
          x <- x[valid]
          y <- y[valid]
          
          n <- length(x)
          
          if (
            n < 4 ||
            length(unique(x)) < 2 ||
            length(unique(y)) < 2
          ) {
            
            return(
              data.frame(
                rho = NA_real_,
                p_value = NA_real_,
                n = n
              )
            )
          }
          
          test <- suppressWarnings(
            cor.test(
              x,
              y,
              method = "spearman",
              exact = FALSE
            )
          )
          
          data.frame(
            rho = unname(test$estimate),
            p_value = test$p.value,
            n = n
          )
        }
      )
    ) %>%
    
    tidyr::unnest(result) %>%
    
    mutate(
      p_FDR = p.adjust(p_value, method = "BH"),
      
      significance = case_when(
        !is.na(p_FDR) & p_FDR < 0.001 ~ "***",
        !is.na(p_FDR) & p_FDR < 0.01  ~ "**",
        !is.na(p_FDR) & p_FDR < 0.05  ~ "*",
        TRUE ~ ""
      ),
      
      protein_label = sub("^nulisa_", "", protein),
      
      correlation_label = case_when(
        !is.na(rho) ~ paste0(round(rho, 2), significance),
        TRUE ~ ""
      )
    )
}
# ============================================================
# 12. CALCULATE BOTH CORRELATION MATRICES
# ============================================================

cor_demographics <- calculate_correlations(
  data = df,
  proteins = nulisa_proteins,
  variables = demographic_variables
)

cor_biomarkers <- calculate_correlations(
  data = df,
  proteins = nulisa_proteins,
  variables = biomarker_columns
)


# ============================================================
# 13. SPLIT PROTEINS INTO TWO ALPHABETICAL HALVES
# ============================================================

n_proteins <- length(nulisa_proteins)

if (n_proteins < 2) {
  stop("At least two exploratory proteins are needed.")
}

split_point <- ceiling(n_proteins / 2)

proteins_part1 <- nulisa_proteins[
  seq_len(split_point)
]

proteins_part2 <- nulisa_proteins[
  seq.int(split_point + 1, n_proteins)
]

cat(
  "\nTotal proteins:", n_proteins,
  "\nPart 1:", length(proteins_part1),
  "\nPart 2:", length(proteins_part2),
  "\n"
)

# ============================================================
# 14. VARIABLE LABELS
# ============================================================

labels_demographics <- c(
  "age_bl1" = "Age",
  "sex_female" = "Sex (female)",
  "apoe4_status" = "APOE ε4 carrier"
)

labels_biomarkers <- c(
  "nulisa_pTau181_0" = "p-tau181",
  "nulisa_pTau231_0" = "p-tau231",
  "nulisa_pTau217_0" = "p-tau217",
  "AB42_40_ratio" = "Aβ42/40\n(log2)",
  "nulisa_NEFL_0" = "NfL",
  "nulisa_GFAP_0" = "GFAP",
  "lumipulse_ptau217_0" = "Lumipulse\np-tau217"
)

cat(
  "Final analytical population: N =",
  dplyr::n_distinct(df$ID),
  "\n"
)


# ============================================================
# 15. HEATMAP FUNCTION
# ============================================================

make_correlation_heatmap <- function(
    results,
    proteins,
    variables,
    variable_labels
) {
  
  plot_data <- results %>%
    dplyr::filter(protein %in% proteins) %>%
    dplyr::mutate(
      protein = factor(
        protein,
        levels = rev(proteins)
      ),
      
      variable = factor(
        variable,
        levels = variables
      )
    )
  
  ggplot(
    plot_data,
    aes(
      x = variable,
      y = protein,
      fill = rho
    )
  ) +
    
    geom_tile(
      color = "white",
      linewidth = 0.25
    ) +
    
    geom_text(
      aes(label = significance),
      size = 3.5,
      color = "black",
      fontface = "bold"
    ) +
    
    scale_fill_gradient2(
      low = "#2166AC",
      mid = "white",
      high = "#B2182B",
      midpoint = 0,
      limits = c(-1, 1),
      name = "Spearman\nrho",
      na.value = "grey90"
    ) +
    
    scale_x_discrete(
      labels = variable_labels,
      expand = expansion(add = 0)
    ) +
    
    scale_y_discrete(
      position = "right",
      labels = function(x) {
        sub("_0$", "", sub("^nulisa_", "", x))
      },
      expand = expansion(add = 0)
    ) +
    
    # No coord_fixed(): allows wider cells
    
    theme_minimal(
      base_size = 12
    ) +
    
    theme(
      panel.grid = element_blank(),
      
      axis.title = element_blank(),
      
      axis.text.x = element_text(
        size = 10,
        color = "black",
        face = "plain",
        margin = margin(t = 8)
      ),
      
      axis.text.y = element_text(
        size = 8.5,
        color = "black",
        margin = margin(l = 5)
      ),
      
      axis.ticks = element_blank(),
      
      legend.title = element_text(size = 10),
      
      legend.text = element_text(size = 9),
      
      legend.key.height = grid::unit(2.5, "cm"),
      
      plot.margin = margin(5, 5, 5, 5)
    )
}


# ============================================================
# 16. CREATE FOUR INDIVIDUAL HEATMAPS
# ============================================================

p_demo_1 <- make_correlation_heatmap(
  results = cor_demographics,
  proteins = proteins_part1,
  variables = demographic_variables,
  variable_labels = labels_demographics
)

p_demo_2 <- make_correlation_heatmap(
  results = cor_demographics,
  proteins = proteins_part2,
  variables = demographic_variables,
  variable_labels = labels_demographics
)

p_biomarker_1 <- make_correlation_heatmap(
  results = cor_biomarkers,
  proteins = proteins_part1,
  variables = biomarker_columns,
  variable_labels = labels_biomarkers
)

p_biomarker_2 <- make_correlation_heatmap(
  results = cor_biomarkers,
  proteins = proteins_part2,
  variables = biomarker_columns,
  variable_labels = labels_biomarkers
)


# ============================================================
# 17. COMBINE HALVES SIDE BY SIDE
# ============================================================

library(patchwork)

# The spacer creates a gap between the two heatmaps.
# Increase 0.20 if the protein names are too close
# to the second heatmap.

combined_demographics <- (
  p_demo_1 +
    patchwork::plot_spacer() +
    p_demo_2
) +
  patchwork::plot_layout(
    ncol = 3,
    widths = c(1, 0.20, 1),
    guides = "collect"
  ) &
  theme(
    legend.position = "right"
  )


combined_biomarkers <- (
  p_biomarker_1 +
    patchwork::plot_spacer() +
    p_biomarker_2
) +
  patchwork::plot_layout(
    ncol = 3,
    widths = c(1, 0.20, 1),
    guides = "collect"
  ) &
  theme(
    legend.position = "right"
  )


# ============================================================
# 18. DISPLAY COMBINED FIGURES
# ============================================================

print(combined_demographics)

print(combined_biomarkers)



# ============================================================
# 19. FIGURE DIMENSIONS
# ============================================================

fig_height <- max(
  9,
  max(
    length(proteins_part1),
    length(proteins_part2)
  ) * 0.23
)


# ============================================================
# 20. SAVE DEMOGRAPHIC CORRELATION FIGURE
# ============================================================

ggsave(
  filename = file.path(
    output_path,
    "Supplementary_Age_Sex_APOE4_Combined_FDR.svg"
  ),
  plot = combined_demographics,
  device = svglite::svglite,
  width = 12,
  height = fig_height,
  units = "in",
  bg = "white"
)

ggsave(
  filename = file.path(
    output_path,
    "Supplementary_Age_Sex_APOE4_Combined_FDR.tiff"
  ),
  plot = combined_demographics,
  width = 12,
  height = fig_height,
  units = "in",
  dpi = 600,
  compression = "lzw",
  bg = "white"
)


# ============================================================
# 21. SAVE CORE BIOMARKER CORRELATION FIGURE
# ============================================================

ggsave(
  filename = file.path(
    output_path,
    "Supplementary_AD_Biomarkers_Combined_FDR.svg"
  ),
  plot = combined_biomarkers,
  device = svglite::svglite,
  width = 16,
  height = fig_height,
  units = "in",
  bg = "white"
)

ggsave(
  filename = file.path(
    output_path,
    "Supplementary_AD_Biomarkers_Combined_FDR.tiff"
  ),
  plot = combined_biomarkers,
  width = 16,
  height = fig_height,
  units = "in",
  dpi = 600,
  compression = "lzw",
  bg = "white"
)


# ============================================================
# 22. EXPORT COMPLETE CORRELATION RESULTS
# ============================================================

readr::write_csv(
  cor_demographics,
  file.path(
    output_path,
    "Correlations_Age_Sex_APOE4_ALL.csv"
  )
)

readr::write_csv(
  cor_biomarkers,
  file.path(
    output_path,
    "Correlations_AD_Biomarkers_ALL.csv"
  )
)

