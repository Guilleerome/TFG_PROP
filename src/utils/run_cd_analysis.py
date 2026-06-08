"""
run_cd_analysis.py
Ejecuta el análisis de distancia crítica sobre los resultados del experimento.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from io_instances import instances_reader as ir
from algorithms import run_algorithm
from utils.critical_distance import plot_cd_diagram

# ── Configuraciones a comparar ────────────────────────────────────────────
CONFIGS = {
    "GRASP_manual": {
        "algorithm": "grasp", "constructor": "random_greedy_by_row",
        "alpha": 0.75, "sample_size": 40,
        "ls_sequence": ["first_move_swap", "best_move"], "ls_sample_size": 500,
    },
    "GRASP_manual_2": {
        "algorithm": "grasp", "constructor": "random_greedy_by_row",
        "alpha": 0.75, "sample_size": 40,
        "ls_sequence": ["best_move_swap", "best_move"], "ls_sample_size": 500,
    },
    "BVNS_irace": {
        "algorithm": "bvns", "constructor": "global_score_ordering_random",
        "alpha": 0.6091, "sample_size": 40,
        "ls_sequence": ["first_move_swap", "best_move"], "ls_sample_size": 755,
    },
    "Random": {
        "algorithm": "grasp", "constructor": "random", "alpha": 1.0,
    }
}

ITERATIONS   = 10
INSTANCES_FOLDER = "selected_instances"

def main():
    plants = ir.read_instances(INSTANCES_FOLDER)
    print(f"Instancias: {len(plants)}")
    print(f"Configuraciones: {list(CONFIGS.keys())}")
    print(f"Repeticiones: {ITERATIONS}\n")

    # Construir matriz resultados: filas=instancias, columnas=configs
    results = {cfg_name: [] for cfg_name in CONFIGS}

    for plant in plants:
        print(f"  Procesando {plant.name}...")
        for cfg_name, cfg in CONFIGS.items():
            costs = []
            for _ in range(ITERATIONS):
                sol = run_algorithm(
                    algorithm_name=cfg["algorithm"],
                    plant=plant,
                    constructor_name=cfg["constructor"],
                    alpha=cfg["alpha"],
                    sample_size=cfg.get("sample_size", 40),
                    ls_sequence=cfg.get("ls_sequence", []),
                    ls_sample_size=cfg.get("ls_sample_size", 500),
                )
                costs.append(sol.cost)
            results[cfg_name].append(np.mean(costs))

    results_df = pd.DataFrame(results, index=[p.name for p in plants])
    print("\nMatriz de resultados (coste medio por instancia):")
    print(results_df.round(2).to_string())

    # Guardar resultados
    results_dir = Path(__file__).resolve().parents[2] / "results"
    results_dir.mkdir(exist_ok=True)
    results_df.to_csv(results_dir / "cd_results3.csv")
    print(f"\nCSV guardado en {results_dir / 'cd_results3.csv'}")

    # Generar CD diagram
    mean_ranks, cd, p_value = plot_cd_diagram(
        results_df,
        title=f"Critical Distance — Nemenyi (α=0.05, {ITERATIONS} reps)",
        alpha=0.05,
        output_path=str(results_dir / "cd_diagram3.png")
    )

if __name__ == "__main__":
    main()