library(irace)
setwd("W:/CosasUni/TFG_PROP/src/irace")
load("irace.Rdata")

# Modificar directamente el ID 480 con los valores de tu config manual
# (grasp + random_greedy_by_row + alpha=0.75 + sample_size=40 +
#  ls1=first_move_swap + ls2=best_move + ls_sample_size=500)
idx <- which(iraceResults$allConfigurations$.ID. == 1)

iraceResults$allConfigurations[idx, "alpha"]          <- 0.75
iraceResults$allConfigurations[idx, "sample_size"]    <- 40L
iraceResults$allConfigurations[idx, "ls1"]            <- "first_move_swap"
iraceResults$allConfigurations[idx, "ls2"]            <- "best_move"
iraceResults$allConfigurations[idx, "ls_sample_size"] <- 500L
iraceResults$allConfigurations[idx, "constructor"]    <- "random_greedy_by_row"
iraceResults$allConfigurations[idx, "algorithm"]      <- "grasp"

cat("Config SRC modificada (ID 1):\n")
print(iraceResults$allConfigurations[idx, ])

cat("\nConfig TARGET (mejor irace, ID 359):\n")
print(iraceResults$allConfigurations[
  iraceResults$allConfigurations$.ID. == 661, ])

ablation_result <- ablation(
  iraceResults    = iraceResults,
  src             = 1L,
  target          = 661L,
  type            = "full",
  nrep            = 3L,
  seed            = 123456L,
  ablationLogFile = "log-ablation.Rdata"
)

cat("\n=== RESULTADO ===\n")
print(ablation_result)