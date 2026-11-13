

# Helper to safely extract one term
get_term_value <- function(tbl, term_name, value_col) {
  x <- tbl %>% filter(term == term_name) %>% pull({{ value_col }})
  if (length(x) == 0) NA_real_ else x
}


