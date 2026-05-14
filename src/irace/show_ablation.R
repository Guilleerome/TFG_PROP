library(irace)
setwd("W:/CosasUni/TFG_PROP/src/irace")
load("log-ablation.Rdata")

# ── Plot ──────────────────────────────────────────────────────────────────
pdf("ablation_plot.pdf", width=10, height=6)
plotAblation(ablation_result)
dev.off()
cat("Plot guardado en ablation_plot.pdf\n")

# ── Tabla de ruta ─────────────────────────────────────────────────────────
traj_ids   <- as.integer(ablation_result$trajectory)
configs    <- ablation_result$allConfigurations
exp_matrix <- ablation_result$experiments
param_cols <- setdiff(colnames(configs), c(".ID.", ".PARENT."))

cat(sprintf("\n%-5s %-22s %-25s %-25s %-12s\n",
            "Paso", "Parámetro cambiado", "Valor anterior",
            "Valor nuevo", "Coste medio"))
cat(strrep("-", 92), "\n")

for (i in 2:length(traj_ids)) {
  prev_id  <- traj_ids[i-1]
  curr_id  <- traj_ids[i]
  prev_cfg <- configs[which(configs$.ID. == prev_id), ]
  curr_cfg <- configs[which(configs$.ID. == curr_id), ]

  if (nrow(prev_cfg) == 0 || nrow(curr_cfg) == 0) next

  col_name  <- as.character(curr_id)
  mean_cost <- if (col_name %in% colnames(exp_matrix))
    mean(exp_matrix[, col_name], na.rm = TRUE) else NA

  changed_params <- c()
  for (param in param_cols) {
    prev_str <- ifelse(is.na(prev_cfg[[param]]), "NA", as.character(prev_cfg[[param]]))
    curr_str <- ifelse(is.na(curr_cfg[[param]]), "NA", as.character(curr_cfg[[param]]))
    if (prev_str != curr_str) changed_params <- c(changed_params, param)
  }

  if (length(changed_params) >= 1) {
    param    <- changed_params[1]
    prev_str <- ifelse(is.na(prev_cfg[[param]]), "NA", as.character(prev_cfg[[param]]))
    curr_str <- ifelse(is.na(curr_cfg[[param]]), "NA", as.character(curr_cfg[[param]]))
    cat(sprintf("%-5d %-22s %-25s %-25s %-12.2f\n",
                i-1, param, prev_str, curr_str,
                ifelse(is.na(mean_cost), 0, mean_cost)))
  }
}

# ── Costes origen / destino ───────────────────────────────────────────────
src_id   <- traj_ids[1]
tgt_id   <- traj_ids[length(traj_ids)]
src_cost <- mean(exp_matrix[, as.character(src_id)], na.rm = TRUE)
tgt_cost <- mean(exp_matrix[, as.character(tgt_id)], na.rm = TRUE)

cat(strrep("-", 92), "\n")
cat(sprintf("Coste config ORIGEN  (ID %d — manual):       %.2f\n", src_id, src_cost))
cat(sprintf("Coste config DESTINO (ID %d — mejor irace): %.2f\n", tgt_id, tgt_cost))
cat(sprintf("Mejora total:                                 %.2f (%.1f%%)\n",
            src_cost - tgt_cost,
            100 * (src_cost - tgt_cost) / src_cost))

cat("\n=== MEJOR CONFIGURACIÓN FINAL ===\n")
print(ablation_result$best)