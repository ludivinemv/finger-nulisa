
# Function to perform ANOVA using linear model
perform_anova <- function(data, protein, group_var) {
  formula <- as.formula(paste(protein, "~", group_var))
  model <- lm(formula, data = data)
  anova_result <- anova(model)
  tidy_anova <- tidy(anova_result)
  return(tidy_anova)
}

# Function to calculate mean and std for each group
calculate_mean_std <- function(data, protein, group_var) {
  mean_std <- data %>%
    group_by(!!sym(group_var)) %>%
    summarise(
      mean = mean(!!sym(protein), na.rm = TRUE),
      std = sd(!!sym(protein), na.rm = TRUE)
    ) %>%
    mutate(
      mean_std = sprintf("%.2f (%.2f)", mean, std)
    ) %>%
    select(!!sym(group_var), mean_std)
  return(mean_std)
}

# correlation matrix
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




get_linear_combo <- function(model, terms) {
  b <- lme4::fixef(model)
  V <- as.matrix(vcov(model))
  
  L <- rep(0, length(b))
  names(L) <- names(b)
  
  for (tm in terms) {
    if (tm %in% names(L)) L[tm] <- 1
  }
  
  est <- sum(L * b)
  se  <- sqrt(as.numeric(t(L) %*% V %*% L))
  z   <- est / se
  p   <- 2 * pnorm(abs(z), lower.tail = FALSE)
  
  tibble(
    estimate = est,
    std.error = se,
    p.value = p,
    ci_low = est - 1.96 * se,
    ci_high = est + 1.96 * se
  )
}