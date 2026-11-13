

create_top25_tables <- function(results,
                                output_path,
                                folder,
                                method,
                                p
) {
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
  
  # Top 25 with p <= 0.10
  top25_df <- results_filtered %>%
    dplyr::filter(.data[[p]] <= 0.10) %>%
    dplyr::arrange(.data[[p]]) %>%
    dplyr::slice_head(n = 25)
  
  if (!"Protein" %in% names(top25_df)) {
    names(top25_df)[1] <- "Protein"
  }
  
  readr::write_csv(
    top25_df,
    file.path(
      output_path,
      folder,
      paste0("top25_", method, "_pval_0.1_excluding_core.csv")
    )
  )
  
  # Top 25 with p <= 0.05
  top25_df_low <- results_filtered %>%
    dplyr::filter(.data[[p]] <= 0.05) %>%
    dplyr::arrange(.data[[p]]) %>%
    dplyr::slice_head(n = 25)
  
  if (!"Protein" %in% names(top25_df_low)) {
    names(top25_df_low)[1] <- "Protein"
  }
  
  readr::write_csv(
    top25_df_low,
    file.path(
      output_path,
      folder,
      paste0("top25_", method, "_pval_0.05_excluding_core.csv")
    )
  )
  
  invisible(list(
    top25_df = top25_df,
    top25_df_low = top25_df_low
  ))
}




plot_volcano <- function(output_path,
                         folder,
                         output_file_name,
                         title,
                         results_filtered,
                         top25_df,
                         p, 
                         beta) {
  p_fdr <- paste0(p," FDR")
  if (!(p_fdr %in% colnames(results_filtered))){
    results_filtered[p_fdr] <- 1
  }
  # ...........................................................................
  # Prepare data
  # ...........................................................................
  volcano_df <- results_filtered %>%
    dplyr::filter(!is.na(.data[[beta]]), !is.na(.data[[p]]), .data[[p]] > 0) %>%
    dplyr::mutate(
      neglog10p = -log10(.data[[p]]),
      status = dplyr::case_when(
        .data[[p_fdr]] < 0.05 & .data[[beta]] < 0 ~ "Negative, FDR p < 0.05",
        .data[[p]] < 0.05 & .data[[beta]] < 0 ~ "Negative, p < 0.05",
        .data[[p]] < 0.05 & .data[[beta]]  > 0 ~ "Positive, p < 0.05",
        .data[[p]] < 0.05 & .data[[beta]]  > 0 ~ "Positive, FDR p < 0.05",
        .data[[p]] < 0.10 & .data[[beta]]  < 0 ~ "Negative, 0.05 <= p < 0.10",
        .data[[p]] < 0.10 & .data[[beta]]  > 0 ~ "Positive, 0.05 <= p < 0.10",
        TRUE ~ "Not significant"
      )
    )
  
  # ...........................................................................
  # Labels: proteins listed in top25_df_low$Proteins
  # ...........................................................................
  label_df <- volcano_df %>%
    dplyr::filter(biomarker %in% top25_df$Protein)
  
  # ...........................................................................
  # Color palette
  # ...........................................................................
  volcano_colors <- c(
    "Negative, FDR p < 0.05" = "#9766CC",
    "Negative, p < 0.05" = "#CC6677",
    "Negative, 0.05 <= p < 0.10" = "#E7A1B0",
    "Not significant" = "#9A9A9A",
    "Positive, 0.05 <= p < 0.10" = "#88CCAA",
    "Positive, p < 0.05" = "#228833",
    "Positive, FDR p < 0.05" = "#224288"
  )
  
  # ...........................................................................
  # Plot
  # ...........................................................................
  p_volcano <- ggplot(volcano_df,
                      aes(x = .data[[beta]], y = neglog10p, color = status)) +
    geom_hline(
      yintercept = -log10(0.05),
      linetype = "dashed",
      linewidth = 0.5,
      color = "grey45"
    ) +
    geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.4,
      color = "grey60"
    ) +
    geom_point(size = 2.8, alpha = 0.9) +
    ggrepel::geom_text_repel(
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
      title = title,
      x = expression(beta ~ "coefficient (per 1 SD higher biomarker, NPQ)"),
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
  
  print(p_volcano)
  
  # ...........................................................................
  # Save figure and data
  # ...........................................................................
  ggsave(
    filename = file.path(output_path, folder, paste0(output_file_name, ".png")),
    plot = p_volcano,
    width = 9,
    height = 7,
    dpi = 600
  )
  
  ggsave(
    filename = file.path(output_path, folder, paste0(output_file_name, ".svg")),
    plot = p_volcano,
    device = svglite::svglite,
    width = 9,
    height = 7
  )
  
  readr::write_csv(
    volcano_df,
    file.path(output_path, folder, paste0(output_file_name, ".csv"))
  )
  
  invisible(list(
    plot = p_volcano,
    data = volcano_df,
    labels = label_df
  ))
}
