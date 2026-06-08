"""
critical_distance.py
Genera el Critical Distance diagram basado en el test de Nemenyi.
Compara múltiples algoritmos sobre múltiples instancias.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from scipy import stats
from scipy.stats import friedmanchisquare


def compute_rankings(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Dado un DataFrame (filas=instancias, columnas=algoritmos, valores=costes),
    devuelve el DataFrame de rankings (1=mejor).
    """
    return results_df.rank(axis=1, ascending=True)


def friedman_test(results_df: pd.DataFrame):
    """
    Aplica el test de Friedman para ver si hay diferencias significativas.
    Debe ser significativo (p < 0.05) antes de aplicar Nemenyi.
    """
    groups = [results_df[col].values for col in results_df.columns]
    stat, p_value = friedmanchisquare(*groups)
    return stat, p_value


def nemenyi_test(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el test post-hoc de Nemenyi.
    Devuelve matriz de p-values.
    """
    return sp.posthoc_nemenyi_friedman(results_df)


def critical_distance(n_instances: int, n_algorithms: int,
                      alpha: float = 0.05) -> float:
    """
    Calcula la distancia crítica de Nemenyi.
    """
    # Valores críticos de la distribución de rangos studentizada
    q_alpha = {
        0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
               6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164},
        0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459,
               6: 2.589, 7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920}
    }
    q = q_alpha[alpha].get(n_algorithms, 2.569)
    cd = q * np.sqrt(n_algorithms * (n_algorithms + 1) / (6 * n_instances))
    return cd


def plot_cd_diagram(results_df: pd.DataFrame,
                    title: str = "Critical Distance Diagram (Nemenyi)",
                    alpha: float = 0.05,
                    output_path: str = None):
    n_instances  = len(results_df)
    n_algorithms = len(results_df.columns)
    alg_names    = list(results_df.columns)

    # ── Friedman ──────────────────────────────────────────────────────────
    stat, p_value = friedman_test(results_df)
    print(f"\nTest de Friedman: stat={stat:.4f}, p={p_value:.6f}")
    if p_value >= alpha:
        print(f"  AVISO: p={p_value:.4f} >= {alpha}. Sin diferencias significativas.")
    else:
        print(f"  Diferencias significativas (p={p_value:.4f}). Aplicando Nemenyi.")

    # ── Rankings medios ───────────────────────────────────────────────────
    rankings   = compute_rankings(results_df)
    mean_ranks = rankings.mean(axis=0).sort_values()
    print("\nRankings medios:")
    for alg, rank in mean_ranks.items():
        print(f"  {alg:<35} {rank:.3f}")

    # ── CD ────────────────────────────────────────────────────────────────
    cd = critical_distance(n_instances, n_algorithms, alpha)
    print(f"\nDistancia crítica (CD={cd:.4f}, alpha={alpha})")

    # ── Nemenyi p-values ──────────────────────────────────────────────────
    p_matrix = nemenyi_test(results_df)
    p_matrix.index   = alg_names
    p_matrix.columns = alg_names

    # ── Plot ──────────────────────────────────────────────────────────────
    fig_height = max(4, n_algorithms * 0.6 + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.set_xlim(0.3, n_algorithms + 0.7)
    ax.set_ylim(-0.5, n_algorithms + 2.0)
    ax.invert_xaxis()

    # Eje X arriba
    ax.xaxis.set_tick_params(top=True, labeltop=True,
                              bottom=False, labelbottom=False)
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Ranking medio", labelpad=8, fontsize=11)
    ax.set_yticks([])
    for spine in ["left", "right", "bottom"]:
        ax.spines[spine].set_visible(False)

    # Línea horizontal principal — se define ANTES de usarla
    y_main = n_algorithms + 0.3
    ax.axhline(y=y_main, color="black", linewidth=1.5)

    # CD bar — esquina superior izquierda (rankings altos = peor)
    cd_x_start = n_algorithms + 0.3
    ax.annotate("",
                xy=(cd_x_start - cd, y_main + 0.6),
                xytext=(cd_x_start, y_main + 0.6),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(cd_x_start - cd / 2, y_main + 0.95,
            f"CD = {cd:.2f}", ha="center", fontsize=9)

    # Algoritmos: mitad izquierda / mitad derecha
    sorted_names  = mean_ranks.index.tolist()
    sorted_values = mean_ranks.values.tolist()
    mid = (n_algorithms + 1) // 2

    left_algs  = list(zip(sorted_names[:mid],  sorted_values[:mid]))
    right_algs = list(zip(sorted_names[mid:],  sorted_values[mid:]))

    def draw_alg(name, rank, side, pos_idx, total):
        y_label = (total - 1) - pos_idx * (total / max(total, 1)) + 0.3
        y_label = max(y_label, 0)
        ax.plot([rank, rank], [y_main, y_label],
                color="black", lw=0.8, ls="--", alpha=0.6)
        ax.plot(rank, y_main, "o", color="black", markersize=5)
        ha = "right" if side == "left" else "left"
        offset = 0.06 if side == "left" else -0.06
        ax.text(rank + offset, y_label, name,
                ha=ha, va="center", fontsize=9)

    for i, (name, rank) in enumerate(left_algs):
        draw_alg(name, rank, "left", i, mid)
    for i, (name, rank) in enumerate(right_algs):
        draw_alg(name, rank, "right", i, len(right_algs))

    # Grupos no significativamente diferentes
    drawn = set()
    for i, alg_i in enumerate(sorted_names):
        group = [alg_i]
        for alg_j in sorted_names:
            if alg_i != alg_j and p_matrix.loc[alg_i, alg_j] > alpha:
                group.append(alg_j)
        if len(group) > 1:
            key = frozenset(group)
            if key not in drawn:
                drawn.add(key)
                group_ranks = [mean_ranks[a] for a in group]
                y_bar = y_main - 0.25
                ax.plot([max(group_ranks), min(group_ranks)],
                        [y_bar, y_bar],
                        color="black", lw=5, alpha=0.35,
                        solid_capstyle="round")

    ax.set_title(title, pad=25, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot guardado en {output_path}")
    else:
        plt.show()

    return mean_ranks, cd, p_value