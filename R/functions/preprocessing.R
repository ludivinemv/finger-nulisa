

# ..............................................................................
## Biomarker transformation ----------------------------------------------------
# ..............................................................................
# NULISA values are NPQ, which are ALREADY log2-transformed 
# Lumipulse pTau217 is a concentration (pg/mL) -> log-transformed.
# Windosorisation is applied to all biomarkers 

# Windosorisation
winsorize <- function(x, p = 0.01) {
  q <- quantile(x, probs = c(p, 1 - p), na.rm = TRUE, names = FALSE)
  x[x < q[1]] <- q[1]
  x[x > q[2]] <- q[2]
  x
}

# Safe z-score function
zscore_safe <- function(x) {
  s <- sd(x, na.rm = TRUE)
  if (is.na(s) || s == 0) {
    rep(NA_real_, length(x))
  } else {
    as.numeric((x - mean(x, na.rm = TRUE)) / s)
  }
}





rename_proteins <- function(df, col_name) {
  
  x <- df[[col_name]]
  
  # Remove "nulisa_" at the beginning, except for nulisa_ptau217_0
  x <- ifelse(
    x == "nulisa_ptau217_0",
    x,
    sub("^nulisa_", "", x)
  )
  
  # Remove "_0" at the end
  x <- sub("_0$", "", x)
  
  # Rename cognition variables
  x <- dplyr::recode(
    x,
    "ztotal_cognition_change" = "Global Cognition Change",
    "ztotal_cognition_baseline" = "Global Cognition Baseline",
    "lumipulse_ptau217" = "Lumipulse ptau217"
  )
  
  df[[col_name]] <- x
  
  df
}




needs_log <- function(bm_name) grepl("lumipulse", bm_name)

log_transformation <- function(x, bm_name) {
  if (needs_log(bm_name)) log(x) else x
}

# replace zeros with half minimum positive
replace_zero_halfmin <- function(x) {
  
  min_pos <- min(x[x > 0], na.rm = TRUE)
  
  x_new <- x
  x_new[x_new == 0] <- min_pos / 2
  
  x_new
}

# ..............................................................................
## Biomarkers exclusion  ----------------------------------------------------
# ..............................................................................
drop_excluded <- function(bm_names) {
  short <- sub("_0$", "", sub("^nulisa_", "", bm_names))
  bm_names[!short %in% EXCLUDE_FROM_MODELS]
}