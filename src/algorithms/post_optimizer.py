from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Optional

from models.plant import Plant
from models.solution import Solution
from improvers.local_search import copy_disposition
from .perturbations import ALL_PERTURBATIONS
from .registry import LOCAL_SEARCHES


# ── Todas las LS disponibles ───────────────────────────────────────────────
ALL_LS_SEQUENCES = [
    ['first_move_swap', 'best_move'],
    ['best_move_swap', 'first_move'],
    ['first_move_swap', 'best_move', 'first_move'],
    ['best_move_swap', 'best_move'],
    ['first_move', 'best_move'],
]


def _apply_perturbation_by_name(disposition: list[list[int]], name: str) -> list[list[int]]:
    return ALL_PERTURBATIONS[name](disposition)


def _worker(args: tuple) -> tuple[str, str, Solution]:
    """
    Ejecuta una combinación perturbación + LS sobre una disposición base.
    Devuelve (perturb_name, ls_label, solution).
    """
    plant, base_disposition, base_cost, perturb_name, ls_sequence, ls_sample_size = args

    # 1. Partir de la disposición base
    disp = copy_disposition(base_disposition)
    cost = base_cost
    sol = Solution(plant, disp, cost)

    # 2. Perturbar
    perturbed_disp = _apply_perturbation_by_name(sol.disposition, perturb_name)
    perturbed_cost = plant.evaluator.evaluate(perturbed_disp)
    sol = Solution(plant, perturbed_disp, perturbed_cost)

    # 3. Aplicar secuencia de LS
    ls_functions = [LOCAL_SEARCHES[name] for name in ls_sequence if name in LOCAL_SEARCHES]
    for ls_func in ls_functions:
        sol = ls_func(sol, ls_sample_size)

    ls_label = '+'.join(ls_sequence)
    return perturb_name, ls_label, sol


def run_from_solution(
    base_solution: Solution,
    ls_sample_size: int = 480,
    perturbations: Optional[list[str]] = None,
    ls_sequences: Optional[list[list[str]]] = None,
    verbose: bool = True,
) -> tuple[Solution, list[dict]]:

    plant = base_solution.plant
    base_disp = base_solution.disposition
    base_cost = base_solution.cost

    perturb_names = perturbations or list(ALL_PERTURBATIONS.keys())
    sequences     = ls_sequences  or ALL_LS_SEQUENCES

    # Generar todas las combinaciones
    combos = list(product(perturb_names, sequences))
    total  = len(combos)

    if verbose:
        print(f"\n[PostOpt] {plant.name} | coste base: {base_cost:.2f}")
        print(f"[PostOpt] Probando {total} combinaciones "
              f"({len(perturb_names)} perturbaciones × {len(sequences)} LS)\n")

    args_list = [
        (plant, base_disp, base_cost, p_name, ls_seq, ls_sample_size)
        for p_name, ls_seq in combos
    ]

    results = []
    best_solution = base_solution

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(_worker, args): args for args in args_list}
        for future in as_completed(futures):
            perturb_name, ls_label, sol = future.result()
            improved = sol.cost < base_cost
            results.append({
                'perturbation': perturb_name,
                'ls_sequence':  ls_label,
                'cost':         sol.cost,
                'improvement':  base_cost - sol.cost,
                'improved':     improved,
                'solution':     sol,
            })
            if sol.cost < best_solution.cost:
                best_solution = sol
                if verbose:
                    print(f"  ✓ MEJORA  [{perturb_name}] + [{ls_label}] "
                          f"→ {sol.cost:.2f}  (Δ={base_cost - sol.cost:.2f})")

    results.sort(key=lambda r: r['cost'])

    if verbose:
        improved_count = sum(1 for r in results if r['improved'])
        print(f"\n[PostOpt] Combinaciones que mejoraron: {improved_count}/{total}")
        print(f"[PostOpt] Mejor coste encontrado:      {best_solution.cost:.2f}")
        if best_solution.cost < base_cost:
            print(f"[PostOpt] Mejora total:                {base_cost - best_solution.cost:.2f}")
        else:
            print(f"[PostOpt] No se encontró mejora sobre el coste base.")
        print("\n[PostOpt] Top 5 combinaciones:")
        for r in results[:5]:
            tag = "✓" if r['improved'] else " "
            print(f"  {tag} [{r['perturbation']}] + [{r['ls_sequence']}] → {r['cost']:.2f}")

    return best_solution, results