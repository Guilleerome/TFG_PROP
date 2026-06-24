from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Optional

from models.plant import Plant
from models.solution import Solution
from improvers.local_search import copy_disposition
from .perturbations import ALL_PERTURBATIONS
from .registry import LOCAL_SEARCHES


# ── Todas las secuencias de LS disponibles ──────────────────────────────────
ALL_LS_SEQUENCES = [
    ['first_move_swap', 'best_move'],
    ['first_move_swap', 'first_move'],
    ['best_move_swap', 'best_move'],
]


def _apply_perturbation_by_name(disposition: list[list[int]], name: str, k: int) -> list[list[int]]:
    return ALL_PERTURBATIONS[name](disposition, k)


def _worker(args: tuple) -> tuple[str, str, Solution]:
    """
    Ejecuta una combinacion perturbacion + LS sobre una disposicion base.
    Devuelve (perturb_name, ls_label, solution).
    """
    plant, base_disposition, base_cost, perturb_name, ls_sequence, ls_sample_size, k = args

    # 1. Partir de la disposicion base
    disp = copy_disposition(base_disposition)
    sol = Solution(plant, disp, base_cost)

    # 2. Perturbar con intensidad k (proporcional al tamaño de la instancia)
    perturbed_disp = _apply_perturbation_by_name(sol.disposition, perturb_name, k)
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
    k_ratio: float = 0.15,
    perturbations: Optional[list[str]] = None,
    ls_sequences: Optional[list[list[str]]] = None,
    max_passes: int = 10,
    verbose: bool = False,
) -> tuple[Solution, list[dict]]:

    plant = base_solution.plant

    perturb_names = perturbations or list(ALL_PERTURBATIONS.keys())
    sequences     = ls_sequences  or ALL_LS_SEQUENCES

    # k proporcional al tamaño de la instancia (nº de facilities a perturbar)
    k = max(1, round(k_ratio * plant.number))

    combos = list(product(perturb_names, sequences))
    combos_per_pass = len(combos)

    if verbose:
        print(f"\n[PostOpt] {plant.name} | coste base: {base_solution.cost:.2f}")
        print(f"[PostOpt] k = {k} (ratio={k_ratio}, n={plant.number}) | "
              f"{combos_per_pass} combinaciones/pasada "
              f"({len(perturb_names)} perturbaciones × {len(sequences)} LS)\n")

    current = base_solution
    all_results: list[dict] = []

    # Un unico pool reutilizado en todas las pasadas
    with ProcessPoolExecutor() as executor:
        pass_num = 0
        improved = True

        while improved and pass_num < max_passes:
            pass_num += 1
            base_disp = current.disposition
            base_cost = current.cost

            args_list = [
                (plant, base_disp, base_cost, p_name, ls_seq, ls_sample_size, k)
                for p_name, ls_seq in combos
            ]

            best_of_pass = current
            futures = {executor.submit(_worker, args): args for args in args_list}
            for future in as_completed(futures):
                perturb_name, ls_label, sol = future.result()
                combo_improved = sol.cost < base_cost
                all_results.append({
                    'pass':         pass_num,
                    'perturbation': perturb_name,
                    'ls_sequence':  ls_label,
                    'cost':         sol.cost,
                    'improvement':  base_cost - sol.cost,
                    'improved':     combo_improved,
                    'solution':     sol,
                })
                if sol.cost < best_of_pass.cost:
                    best_of_pass = sol

            improved = best_of_pass.cost < current.cost
            if improved:
                if verbose:
                    print(f"  ✓ Pasada {pass_num}: MEJORA "
                          f"{current.cost:.2f} → {best_of_pass.cost:.2f} "
                          f"(Δ={current.cost - best_of_pass.cost:.2f})")
                current = best_of_pass        # realimentar la batería
            else:
                if verbose:
                    print(f"  · Pasada {pass_num}: sin mejora, fin del post-opt.")

    if verbose:
        improved_count = sum(1 for r in all_results if r['improved'])
        total = len(all_results)
        print(f"\n[PostOpt] Pasadas ejecutadas:           {pass_num}")
        print(f"[PostOpt] Combinaciones que mejoraron:  {improved_count}/{total}")
        print(f"[PostOpt] Mejor coste encontrado:       {current.cost:.2f}")
        if current.cost < base_solution.cost:
            print(f"[PostOpt] Mejora total:                 {base_solution.cost - current.cost:.2f}")
        else:
            print(f"[PostOpt] No se encontro mejora sobre el coste base.")
        print("\n[PostOpt] Top 5 combinaciones:")
        for r in sorted(all_results, key=lambda x: x['cost'])[:5]:
            tag = "✓" if r['improved'] else " "
            print(f"  {tag} [p{r['pass']}] [{r['perturbation']}] + [{r['ls_sequence']}] → {r['cost']:.2f}")


    return current