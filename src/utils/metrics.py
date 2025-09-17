from src.io_instances import instances_reader as ir
from src.constructors import constructor as construct
import time
from src.improvers import local_search as ls
import xlsxwriter
from typing import Dict, Any, List, Tuple
import numpy as np

# Abreviaturas para búsquedas locales
LS_ABBR = {
    "first_move_swap": "FMS",
    "best_move_swap": "BMS",
    "first_move": "FM",
    "best_move": "BM",
}

class Metrics:
    def __init__(
        self,
        excel_path: str = "experiments_results.xlsx",
        iterations: int = 60,
        plants = None,
        alphas=None
    ):
        if alphas is None:
            alphas = [0.25, 0.5, 0.75, 1.0]
        self.iterations = iterations
        if not excel_path.endswith(".xlsx"):
            excel_path += ".xlsx"
        self.excel_path =  "../results/" + excel_path
        self.alphas = alphas

        if plants is None:
            self.plants = ir.read_instances("small")
        else:
            self.plants = plants
        self.results: Dict[str, Any] = {}

        start = time.time()
        self.run_experiments()
        self.write_results_to_excel()
        end = time.time()
        print(f"Total metrics time: {end - start:.2f}s")

    def get_alias(self, name: str) -> str:
        constructor_aliases = {
            "random": "C₀",
            "guillermo": "C₁",
            "greedy_random_by_row": "C₂",
            "greedy_random_global": "C₃",
            "greedy_random_row_balanced": "C₄",
            "random_greedy_by_row": "C₅",
            "random_greedy_global": "C₆",
            "random_greedy_row_balanced": "C₇",
            "global_score_ordering": "C₈",
            "global_score_ordering_random": "C₉",
        }
        if "(α=" in name:
            base, alpha = name.split("(α=")
            alpha = "(α=" + alpha
            base = base.strip()
        elif "(alpha" in name:
            base, alpha = name.split("(alpha")
            alpha = "(α=" + alpha
            base = base.strip()
        else:
            base, alpha = name.strip(), ""
        return f"{constructor_aliases.get(base, base)} {alpha}".strip()

    def run_experiments(self):
        iteraciones = 60

        for plant in self.plants:
            print(f"Procesando planta: {plant.name}")

            # Tiempos agregados (medias) por metodo (para tablas agregadas)
            times = {"Guillermo": 0.0, "random": 0.0, "greedy_random_by_row": [],
                     "greedy_random_global": [], "greedy_random_row_balanced": [],
                     "random_greedy_by_row": [], "random_greedy_global": [],
                     "random_greedy_row_balanced": [], "global_score_ordering": 0.0,
                     "global_score_ordering_random": []}

            # --- Constructores deterministas con tiempo por ejecución ---
            start_time = time.time()
            guillermo_solution = construct.construct_guillermo(plant)
            guillermo_time = time.time() - start_time
            times["Guillermo"] = guillermo_time

            start_time = time.time()
            global_score_ordering_solution = construct.constructor_global_score_ordering(plant, 0.2)
            gso_time = time.time() - start_time
            times["global_score_ordering"] = gso_time

            # Acumuladores por corrida (guardamos (alpha, solution, build_time))
            greedy_random_by_row_runs: List[Tuple[float, Any, float]] = []
            greedy_random_global_runs: List[Tuple[float, Any, float]] = []
            greedy_random_row_balanced_runs: List[Tuple[float, Any, float]] = []
            random_greedy_by_row_runs: List[Tuple[float, Any, float]] = []
            random_greedy_global_runs: List[Tuple[float, Any, float]] = []
            random_greedy_row_balanced_runs: List[Tuple[float, Any, float]] = []
            gso_random_runs: List[Tuple[float, Any, float]] = []
            random_runs: List[Tuple[Any, float]] = []

            # Acumuladores de tiempo promedio por alpha
            greedy_random_by_row_times = []
            greedy_random_global_times = []
            greedy_random_row_balanced_times = []
            random_greedy_by_row_times = []
            random_greedy_global_times = []
            random_greedy_row_balanced_times = []
            gso_random_times = []

            for alpha in self.alphas:
                t_grbr_acc = 0.0
                t_grg_acc = 0.0
                t_grrb_acc = 0.0
                t_rgb_acc = 0.0
                t_rgg_acc = 0.0
                t_rgrb_acc = 0.0
                t_gsor_acc = 0.0

                for _ in range(iteraciones):
                    # Greedy random by row
                    start_time = time.time()
                    sol_aux = construct.constructor_greedy_random_by_row(plant, alpha)
                    elapsed = time.time() - start_time
                    t_grbr_acc += elapsed
                    greedy_random_by_row_runs.append((alpha, sol_aux, elapsed))

                    # Greedy random global
                    start_time = time.time()
                    sol_aux = construct.constructor_greedy_random_global(plant, alpha)
                    elapsed = time.time() - start_time
                    t_grg_acc += elapsed
                    greedy_random_global_runs.append((alpha, sol_aux, elapsed))

                    # Greedy random row balanced
                    start_time = time.time()
                    sol_aux = construct.constructor_greedy_random_row_balanced(plant, alpha)
                    elapsed = time.time() - start_time
                    t_grrb_acc += elapsed
                    greedy_random_row_balanced_runs.append((alpha, sol_aux, elapsed))

                    # Random Greedy by row
                    start_time = time.time()
                    sol_aux = construct.constructor_random_greedy_by_row(plant, alpha)
                    elapsed = time.time() - start_time
                    t_rgb_acc += elapsed
                    random_greedy_by_row_runs.append((alpha, sol_aux, elapsed))

                    # Random Greedy global
                    start_time = time.time()
                    sol_aux = construct.constructor_random_greedy_global(plant, alpha)
                    elapsed = time.time() - start_time
                    t_rgg_acc += elapsed
                    random_greedy_global_runs.append((alpha, sol_aux, elapsed))

                    # Random Greedy row balanced
                    start_time = time.time()
                    sol_aux = construct.constructor_random_greedy_row_balanced(plant, alpha)
                    elapsed = time.time() - start_time
                    t_rgrb_acc += elapsed
                    random_greedy_row_balanced_runs.append((alpha, sol_aux, elapsed))

                    # Global Score Ordering Random
                    start_time = time.time()
                    sol_aux = construct.constructor_global_score_ordering_random(plant, alfa=alpha)
                    elapsed = time.time() - start_time
                    t_gsor_acc += elapsed
                    gso_random_runs.append((alpha, sol_aux, elapsed))

                greedy_random_by_row_times.append(t_grbr_acc / iteraciones)
                greedy_random_global_times.append(t_grg_acc / iteraciones)
                greedy_random_row_balanced_times.append(t_grrb_acc / iteraciones)
                random_greedy_by_row_times.append(t_rgb_acc / iteraciones)
                random_greedy_global_times.append(t_rgg_acc / iteraciones)
                random_greedy_row_balanced_times.append(t_rgrb_acc / iteraciones)
                gso_random_times.append(t_gsor_acc / iteraciones)

            # Random puro (guardamos tiempo por cada ejecución)
            random_time_acc = 0.0
            for _ in range(self.iterations):
                start_time = time.time()
                sol_aux = construct.construct_random(plant)
                elapsed = time.time() - start_time
                random_time_acc += elapsed
                random_runs.append((sol_aux, elapsed))
            random_avg_time = random_time_acc / iteraciones

            # Guardar medias de tiempo por metodo
            times["random"] = random_avg_time
            times["greedy_random_by_row"] = greedy_random_by_row_times
            times["greedy_random_global"] = greedy_random_global_times
            times["greedy_random_row_balanced"] = greedy_random_row_balanced_times
            times["random_greedy_by_row"] = random_greedy_by_row_times
            times["random_greedy_global"] = random_greedy_global_times
            times["random_greedy_row_balanced"] = random_greedy_row_balanced_times
            times["global_score_ordering_random"] = gso_random_times

            # Métricas de costes para calcular best
            all_solution_costs = [guillermo_solution.cost,
                                  global_score_ordering_solution.cost] + \
                                 [s.cost for (s, _) in random_runs] + \
                                 [s.cost for (_, s, _) in greedy_random_by_row_runs] + \
                                 [s.cost for (_, s, _) in greedy_random_global_runs] + \
                                 [s.cost for (_, s, _) in greedy_random_row_balanced_runs] + \
                                 [s.cost for (_, s, _) in random_greedy_by_row_runs] + \
                                 [s.cost for (_, s, _) in random_greedy_global_runs] + \
                                 [s.cost for (_, s, _) in random_greedy_row_balanced_runs] + \
                                 [s.cost for (_, s, _) in gso_random_runs]

            best_cost = min(all_solution_costs)

            def calculate_std_dev(solutions_costs):
                std_devs = [c / best_cost for c in solutions_costs]
                return float(np.mean(std_devs))

            # Deterministas
            guillermo_avg_cost = guillermo_solution.cost
            guillermo_std_dev = calculate_std_dev([guillermo_solution.cost])
            guillermo_num_bests = 1 if guillermo_solution.cost == best_cost else 0

            gso_avg_cost = global_score_ordering_solution.cost
            gso_std_dev = calculate_std_dev([global_score_ordering_solution.cost])
            gso_num_bests = 1 if global_score_ordering_solution.cost == best_cost else 0

            # Agregadores (promedios por alpha)
            def collect_alpha_stats(runs: List[Tuple[float, Any, float]]):
                avg_costs, std_devs, num_bests = {}, {}, {}
                for alpha in self.alphas:
                    solutions_costs = [sol.cost for (a, sol, _) in runs if a == alpha]
                    if solutions_costs:
                        avg_costs[alpha] = float(np.mean(solutions_costs))
                        std_devs[alpha] = calculate_std_dev(solutions_costs)
                        num_bests[alpha] = sum(1 for c in solutions_costs if c == best_cost)
                return avg_costs, std_devs, num_bests

            grbr_avg, grbr_std, grbr_bests = collect_alpha_stats(greedy_random_by_row_runs)
            grg_avg, grg_std, grg_bests = collect_alpha_stats(greedy_random_global_runs)
            grrb_avg, grrb_std, grrb_bests = collect_alpha_stats(greedy_random_row_balanced_runs)
            rgb_avg, rgb_std, rgb_bests = collect_alpha_stats(random_greedy_by_row_runs)
            rgg_avg, rgg_std, rgg_bests = collect_alpha_stats(random_greedy_global_runs)
            rgrb_avg, rgrb_std, rgrb_bests = collect_alpha_stats(random_greedy_row_balanced_runs)
            gsor_avg, gsor_std, gsor_bests = collect_alpha_stats(gso_random_runs)

            random_costs = [s.cost for (s, _) in random_runs]
            random_avg_cost = float(np.mean(random_costs)) if random_costs else 0.0
            random_std_dev = calculate_std_dev(random_costs) if random_costs else 0.0
            random_num_bests = sum(1 for c in random_costs if c == best_cost)

            # ---- Selección de mejores soluciones iniciales (con tiempo de construcción asociado) ----
            all_solutions = (
                [ (s, "Greedy Random by Row", t) for (a, s, t) in greedy_random_by_row_runs ] +
                [ (s, "Greedy Random Global", t) for (a, s, t) in greedy_random_global_runs ] +
                [ (s, "Greedy Random Row Balanced", t) for (a, s, t) in greedy_random_row_balanced_runs ] +
                [ (s, "Random Greedy by Row", t) for (a, s, t) in random_greedy_by_row_runs ] +
                [ (s, "Random Greedy Global", t) for (a, s, t) in random_greedy_global_runs ] +
                [ (s, "Random Greedy Row Balanced", t) for (a, s, t) in random_greedy_row_balanced_runs ] +
                [ (s, "Global Score Ordering Random", t) for (a, s, t) in gso_random_runs ] +
                [ (s, "Random", t) for (s, t) in random_runs ]
            )
            all_solutions += [(guillermo_solution, "Guillermo", guillermo_time)]
            all_solutions += [(global_score_ordering_solution, "Global Score Ordering", gso_time)]

            all_solutions_sorted = sorted(all_solutions, key=lambda x: x[0].cost)

            best_solutions = []
            ordinal_map = ["1st", "2nd", "3rd"]
            current_rank = 0

            for (sol, origin, build_t) in all_solutions_sorted:
                if not best_solutions or sol.cost == best_solutions[-1][0].cost:
                    if len(best_solutions) < len(ordinal_map):
                        best_solutions.append((sol, origin, ordinal_map[current_rank], build_t))
                else:
                    if current_rank < len(ordinal_map) - 1:
                        current_rank += 1
                        best_solutions.append((sol, origin, ordinal_map[current_rank], build_t))
                if len(best_solutions) == len(ordinal_map):
                    break

            # Garantizar que Guillermo aparece si no está
            if not any(sol == guillermo_solution for (sol, _, _, _) in best_solutions):
                last_pos = ordinal_map[-1]
                best_solutions = [(s, o, rk, bt) for (s, o, rk, bt) in best_solutions if rk != last_pos]
                best_solutions.append((guillermo_solution, "Guillermo", ordinal_map[current_rank], guillermo_time))

            # Estructura de entradas para hoja general y para poder sumar tiempos en LS
            best_initial_entries = [
                {"origin": origin, "rank": rank, "cost": sol.cost, "build_time": build_t,
                 "name": f"{origin} ({rank})"}  # name clave para mapping
                for (sol, origin, rank, build_t) in best_solutions
            ]

            # ---- Búsquedas locales (individuales y combinadas) con total_time = build_time + ls_time ----
            local_search_results: Dict[str, Dict[str, Dict[str, float]]] = {}
            extended_local_search_results: Dict[str, Dict[str, Dict[str, float]]] = {}

            local_search_methods = {
                "first_move_swap": ls.first_move_swap,
                "best_move_swap": ls.best_move_swap,
                "first_move": ls.first_move,
                "best_move": ls.best_move
            }

            # Mapa: nombre de solución inicial -> tiempo de construcción
            ctor_time_by_name = {e["name"]: e["build_time"] for e in best_initial_entries}

            # Individual
            for (sol, origin, rank, build_t) in best_solutions:
                sol_name = f"{origin} ({rank})"
                local_search_results[sol_name] = {}
                for key, method in local_search_methods.items():
                    start_time = time.time()
                    result = method(sol)
                    ls_elapsed = time.time() - start_time
                    total_time = build_t + ls_elapsed
                    local_search_results[sol_name][key] = {
                        "cost": result.cost,
                        "time": ls_elapsed,
                        "total_time": total_time
                    }

            # Combinadas
            valid_combinations = [
                ("first_move", "first_move_swap"),
                ("first_move", "best_move_swap"),
                ("best_move", "first_move_swap"),
                ("best_move", "best_move_swap"),
                ("first_move_swap", "first_move"),
                ("first_move_swap", "best_move"),
                ("best_move_swap", "first_move"),
                ("best_move_swap", "best_move")
            ]
            for (sol, origin, rank, build_t) in best_solutions:
                sol_name = f"{origin} ({rank})"
                extended_local_search_results[sol_name] = {}
                for combo in valid_combinations:
                    start_time = time.time()
                    inter = local_search_methods[combo[0]](sol)
                    final = local_search_methods[combo[1]](inter)
                    ls_elapsed = time.time() - start_time
                    total_time = build_t + ls_elapsed
                    combo_label = f"{LS_ABBR[combo[0]]} \u2192 {LS_ABBR[combo[1]]}"
                    extended_local_search_results[sol_name][combo_label] = {
                        "cost": final.cost,
                        "time": ls_elapsed,
                        "total_time": total_time
                    }

            # ---- Guardar resultados por instancia ----
            # Utilidades para obtener mejor solución y su tiempo de construcción por constructor:
            def best_of_runs(runs_list: List[Tuple[float, Any, float]]) -> Tuple[Any, float]:
                if not runs_list:
                    return None, 0.0
                best_tuple = min(runs_list, key=lambda x: x[1].cost)  # (alpha, sol, time)
                return best_tuple[1], best_tuple[2]

            def best_of_random_runs(runs_list: List[Tuple[Any, float]]) -> Tuple[Any, float]:
                if not runs_list:
                    return None, 0.0
                best_tuple = min(runs_list, key=lambda x: x[0].cost)  # (sol, time)
                return best_tuple[0], best_tuple[1]

            grbr_best, grbr_best_time = best_of_runs(greedy_random_by_row_runs)
            grg_best, grg_best_time = best_of_runs(greedy_random_global_runs)
            grrb_best, grrb_best_time = best_of_runs(greedy_random_row_balanced_runs)
            rgb_best, rgb_best_time = best_of_runs(random_greedy_by_row_runs)
            rgg_best, rgg_best_time = best_of_runs(random_greedy_global_runs)
            rgrb_best, rgrb_best_time = best_of_runs(random_greedy_row_balanced_runs)
            gsor_best, gsor_best_time = best_of_runs(gso_random_runs)
            random_best, random_best_time = best_of_random_runs(random_runs)

            self.results[plant.name] = {
                "constructors": {
                    "guillermo": {
                        "best": guillermo_solution,
                        "best_time": guillermo_time,
                        "average": guillermo_avg_cost,
                        "std_devs": guillermo_std_dev,
                        "num_bests": guillermo_num_bests,
                        "time": times["Guillermo"]
                    },
                    "global_score_ordering": {
                        "best": global_score_ordering_solution,
                        "best_time": gso_time,
                        "average": gso_avg_cost,
                        "std_devs": gso_std_dev,
                        "num_bests": gso_num_bests,
                        "time": times["global_score_ordering"]
                    },
                    "random": {
                        "best": random_best,
                        "best_time": random_best_time,
                        "average": random_avg_cost,
                        "std_devs": random_std_dev,
                        "num_bests": random_num_bests,
                        "time": times["random"]
                    },
                    "greedy_random_by_row": {
                        "best": grbr_best, "best_time": grbr_best_time,
                        "averages": grbr_avg, "std_devs": grbr_std, "num_bests": grbr_bests,
                        "times": times["greedy_random_by_row"]
                    },
                    "greedy_random_global": {
                        "best": grg_best, "best_time": grg_best_time,
                        "averages": grg_avg, "std_devs": grg_std, "num_bests": grg_bests,
                        "times": times["greedy_random_global"]
                    },
                    "greedy_random_row_balanced": {
                        "best": grrb_best, "best_time": grrb_best_time,
                        "averages": grrb_avg, "std_devs": grrb_std, "num_bests": grrb_bests,
                        "times": times["greedy_random_row_balanced"]
                    },
                    "random_greedy_by_row": {
                        "best": rgb_best, "best_time": rgb_best_time,
                        "averages": rgb_avg, "std_devs": rgb_std, "num_bests": rgb_bests,
                        "times": times["random_greedy_by_row"]
                    },
                    "random_greedy_global": {
                        "best": rgg_best, "best_time": rgg_best_time,
                        "averages": rgg_avg, "std_devs": rgg_std, "num_bests": rgg_bests,
                        "times": times["random_greedy_global"]
                    },
                    "random_greedy_row_balanced": {
                        "best": rgrb_best, "best_time": rgrb_best_time,
                        "averages": rgrb_avg, "std_devs": rgrb_std, "num_bests": rgrb_bests,
                        "times": times["random_greedy_row_balanced"]
                    },
                    "global_score_ordering_random": {
                        "best": gsor_best, "best_time": gsor_best_time,
                        "averages": gsor_avg, "std_devs": gsor_std, "num_bests": gsor_bests,
                        "times": times["global_score_ordering_random"]
                    }
                },
                "best_initial": {
                    "entries": best_initial_entries  # contiene build_time
                },
                "local_search": {
                    "individual": local_search_results,        # cada metodo: cost, time, total_time
                    "extended": extended_local_search_results  # cada combo: cost, time, total_time
                }
            }

    def write_results_to_excel(self):
        # Crear el archivo Excel
        workbook = xlsxwriter.Workbook(self.excel_path, {"in_memory": True})

        # Formatos
        header_format = workbook.add_format({"bold": True, "bg_color": "#D7E4BC", "border": 1, "align": "center"})
        cell_format = workbook.add_format({"border": 1})
        cell_center = workbook.add_format({"border": 1, "align": "center"})
        cell_wrap_format = workbook.add_format({"border": 1})
        bold_format = workbook.add_format({"bold": True})
        default_column_width = 25

        # ---------- Hoja 0: Legend ----------
        ws_legend = workbook.add_worksheet("Legend")
        ws_legend.set_column(0, 3, 40)
        ws_legend.write_row(0, 0, ["Constructors (alias)", "Name", "Notes"], header_format)
        constructor_legend = [
            ("C₀", "Random", "Uniform random ordering per row."),
            ("C₁", "Guillermo", "Score = flows & size; reorder pattern."),
            ("C₂", "Greedy Random by Row", "Row-wise GRASP (RCL by cost)."),
            ("C₃", "Greedy Random Global", "Global GRASP across rows."),
            ("C₄", "Greedy Random Row Balanced", "Row balancing + GRASP."),
            ("C₅", "Random Greedy by Row", "Random sample, then greedy insert (row)."),
            ("C₆", "Random Greedy Global", "Random sample, then greedy insert (global)."),
            ("C₇", "Random Greedy Row Balanced", "Balanced variant of random-greedy."),
            ("C₈", "Global Score Ordering", "Score = w·flows + (1-w)·size; best insertion."),
            ("C₉", "Global Score Ordering (Randomized)", "RCL over score; best insertion."),
        ]
        r = 1
        for alias, name, note in constructor_legend:
            ws_legend.write_row(r, 0, [alias, name, note], cell_format); r += 1
        r += 1
        ws_legend.write_row(r, 0, ["Local search abbreviations", "Full name"], header_format); r += 1
        for k, v in {"FMS":"first_move_swap", "BMS":"best_move_swap", "FM":"first_move", "BM":"best_move"}.items():
            ws_legend.write_row(r, 0, [k, v], cell_format); r += 1
        r += 1
        ws_legend.write(r, 0, "α (alpha) controls the RCL threshold in GRASP-like constructors (lower α = greedier).", bold_format)
        ws_legend.freeze_panes(1, 0)

        # ---------- Hoja 1: General Results ----------
        ws_general = workbook.add_worksheet("General Results")
        headers_general = [
            "Instance", f"Constructor {self.get_alias('guillermo')} (Cost)",
            f"Constructor {self.get_alias('random')} (Average)",
            f"Constructor {self.get_alias('greedy_random_by_row (α=0.25)')}", f"Constructor {self.get_alias('greedy_random_by_row (α=0.5)')}",
            f"Constructor {self.get_alias('greedy_random_by_row (α=0.75)')}", f"Constructor {self.get_alias('greedy_random_by_row (α=1.0)')}",
            f"Constructor {self.get_alias('greedy_random_global (α=0.25)')}", f"Constructor {self.get_alias('greedy_random_global (α=0.5)')}",
            f"Constructor {self.get_alias('greedy_random_global (α=0.75)')}", f"Constructor {self.get_alias('greedy_random_global (α=1.0)')}",
            f"Constructor {self.get_alias('greedy_random_row_balanced (α=0.25)')}", f"Constructor {self.get_alias('greedy_random_row_balanced (α=0.5)')}",
            f"Constructor {self.get_alias('greedy_random_row_balanced (α=0.75)')}", f"Constructor {self.get_alias('greedy_random_row_balanced (α=1.0)')}",
            f"Constructor {self.get_alias('random_greedy_by_row (α=0.25)')}", f"Constructor {self.get_alias('random_greedy_by_row (α=0.5)')}",
            f"Constructor {self.get_alias('random_greedy_by_row (α=0.75)')}", f"Constructor {self.get_alias('random_greedy_by_row (α=1.0)')}",
            f"Constructor {self.get_alias('random_greedy_global (α=0.25)')}", f"Constructor {self.get_alias('random_greedy_global (α=0.5)')}",
            f"Constructor {self.get_alias('random_greedy_global (α=0.75)')}", f"Constructor {self.get_alias('random_greedy_global (α=1.0)')}",
            f"Constructor {self.get_alias('random_greedy_row_balanced (α=0.25)')}", f"Constructor {self.get_alias('random_greedy_row_balanced (α=0.5)')}",
            f"Constructor {self.get_alias('random_greedy_row_balanced (α=0.75)')}", f"Constructor {self.get_alias('random_greedy_row_balanced (α=1.0)')}",
            f"Constructor {self.get_alias('global_score_ordering')}",
            f"Constructor {self.get_alias('global_score_ordering_random (α=0.25)')}", f"Constructor {self.get_alias('global_score_ordering_random (α=0.5)')}",
            f"Constructor {self.get_alias('global_score_ordering_random (α=0.75)')}", f"Constructor {self.get_alias('global_score_ordering_random (α=1.0)')}",
            "Best Initial (Top1)", "Best Initial (Top2)", "Best Initial (Top3)"
        ]
        ws_general.write_row(0, 0, headers_general, header_format)
        ws_general.set_column(0, len(headers_general) - 1, default_column_width)
        ws_general.freeze_panes(1, 0)

        for row_idx, (plant_name, plant_results) in enumerate(self.results.items(), start=1):
            bi_entries = plant_results["best_initial"]["entries"]
            def bi_cell(i):
                if i < len(bi_entries):
                    e = bi_entries[i]
                    return f"{e['origin']} — {e['cost']} (build {e['build_time']:.4f}s)"
                return "N/A"

            ws_general.write_row(
                row_idx, 0,
                [
                    plant_name,
                    plant_results["constructors"]["guillermo"]["average"],
                    plant_results["constructors"]["random"]["average"],
                    plant_results["constructors"]["greedy_random_by_row"]["averages"].get(0.25, "N/A"),
                    plant_results["constructors"]["greedy_random_by_row"]["averages"].get(0.5, "N/A"),
                    plant_results["constructors"]["greedy_random_by_row"]["averages"].get(0.75, "N/A"),
                    plant_results["constructors"]["greedy_random_by_row"]["averages"].get(1.0, "N/A"),
                    plant_results["constructors"]["greedy_random_global"]["averages"].get(0.25, "N/A"),
                    plant_results["constructors"]["greedy_random_global"]["averages"].get(0.5, "N/A"),
                    plant_results["constructors"]["greedy_random_global"]["averages"].get(0.75, "N/A"),
                    plant_results["constructors"]["greedy_random_global"]["averages"].get(1.0, "N/A"),
                    plant_results["constructors"]["greedy_random_row_balanced"]["averages"].get(0.25, "N/A"),
                    plant_results["constructors"]["greedy_random_row_balanced"]["averages"].get(0.5, "N/A"),
                    plant_results["constructors"]["greedy_random_row_balanced"]["averages"].get(0.75, "N/A"),
                    plant_results["constructors"]["greedy_random_row_balanced"]["averages"].get(1.0, "N/A"),
                    plant_results["constructors"]["random_greedy_by_row"]["averages"].get(0.25, "N/A"),
                    plant_results["constructors"]["random_greedy_by_row"]["averages"].get(0.5, "N/A"),
                    plant_results["constructors"]["random_greedy_by_row"]["averages"].get(0.75, "N/A"),
                    plant_results["constructors"]["random_greedy_by_row"]["averages"].get(1.0, "N/A"),
                    plant_results["constructors"]["random_greedy_global"]["averages"].get(0.25, "N/A"),
                    plant_results["constructors"]["random_greedy_global"]["averages"].get(0.5, "N/A"),
                    plant_results["constructors"]["random_greedy_global"]["averages"].get(0.75, "N/A"),
                    plant_results["constructors"]["random_greedy_global"]["averages"].get(1.0, "N/A"),
                    plant_results["constructors"]["random_greedy_row_balanced"]["averages"].get(0.25, "N/A"),
                    plant_results["constructors"]["random_greedy_row_balanced"]["averages"].get(0.5, "N/A"),
                    plant_results["constructors"]["random_greedy_row_balanced"]["averages"].get(0.75, "N/A"),
                    plant_results["constructors"]["random_greedy_row_balanced"]["averages"].get(1.0, "N/A"),
                    plant_results["constructors"]["global_score_ordering"]["average"],
                    plant_results["constructors"]["global_score_ordering_random"]["averages"].get(0.25, "N/A"),
                    plant_results["constructors"]["global_score_ordering_random"]["averages"].get(0.5, "N/A"),
                    plant_results["constructors"]["global_score_ordering_random"]["averages"].get(0.75, "N/A"),
                    plant_results["constructors"]["global_score_ordering_random"]["averages"].get(1.0, "N/A"),
                    bi_cell(0), bi_cell(1), bi_cell(2)
                ],
                cell_format
            )

        # ---------- Hoja 2: Best Constructor (con tiempo del mejor) ----------
        ws_best_ctor = workbook.add_worksheet("Best Constructor")
        constructors = ["guillermo", "random", "greedy_random_by_row", "greedy_random_global",
                        "greedy_random_row_balanced", "random_greedy_by_row", "random_greedy_global",
                        "random_greedy_row_balanced", "global_score_ordering", "global_score_ordering_random"]

        # Duplicamos columnas: Cost y Time por constructor
        headers = ["Instance"]
        for c in constructors:
            headers += [f"{self.get_alias(c)} (Cost)", f"{self.get_alias(c)} (Time)"]
        headers += ["Best (Cost)"]
        ws_best_ctor.write_row(0, 0, headers, header_format)
        ws_best_ctor.set_column(0, len(headers) - 1, 15)
        ws_best_ctor.freeze_panes(1, 0)

        row = 1
        for instance, data in self.results.items():
            row_values = [instance]
            best_value = float("inf")
            for c in constructors:
                best_sol = data["constructors"][c]["best"]
                best_time = data["constructors"][c].get("best_time", 0.0)
                cost_val = best_sol.cost if best_sol is not None else "N/A"
                row_values += [cost_val, best_time]
                if isinstance(cost_val, (int, float)) and cost_val < best_value:
                    best_value = cost_val
            row_values += [best_value if best_value < float("inf") else "N/A"]

            # Escribir y resaltar mínimos de Cost (no de Time)
            col = 1
            min_cost_cols = []
            # Identificar las columnas de coste para highlight
            costs_for_min = []
            for c in constructors:
                costs_for_min.append(data["constructors"][c]["best"].cost if data["constructors"][c]["best"] is not None else float("inf"))
            min_cost = min(costs_for_min) if costs_for_min else float("inf")

            ws_best_ctor.write(row, 0, instance, cell_center)
            idx_cost_cell = 0
            for c_idx, c in enumerate(constructors):
                cost = costs_for_min[c_idx]
                time_val = data["constructors"][c].get("best_time", 0.0)
                if cost == min_cost:
                    ws_best_ctor.write(row, col, cost, workbook.add_format({"border":1,"align":"center","bold":True,"bg_color":"#FFD700"}))
                else:
                    ws_best_ctor.write(row, col, cost if cost < float("inf") else "N/A", cell_center)
                ws_best_ctor.write(row, col+1, time_val, cell_center)
                col += 2
            ws_best_ctor.write(row, col, min_cost if min_cost < float("inf") else "N/A", cell_center)
            row += 1

        # ---------- Hoja 3: Local Searches (con Total Time) ----------
        ws_local = workbook.add_worksheet("Local Searches")
        headers_local = ["Instance", "Solution", "Method", "Cost", "Time", "Total Time"]
        ws_local.write_row(0, 0, headers_local, header_format)
        ws_local.set_column(0, len(headers_local) - 1, default_column_width)
        ws_local.freeze_panes(1, 0)

        r = 1
        for plant_name, plant_results in self.results.items():
            for solution_id, search_results in plant_results["local_search"]["individual"].items():
                solution_id_alias = self.get_alias(solution_id)
                for method, metrics in search_results.items():
                    method_label = LS_ABBR.get(method, method)
                    ws_local.write_row(
                        r, 0,
                        [
                            plant_name,
                            solution_id_alias,
                            method_label,
                            metrics.get("cost", "N/A"),
                            metrics.get("time", "N/A"),
                            metrics.get("total_time", "N/A")
                        ],
                        cell_wrap_format
                    )
                    r += 1

        # ---------- Hoja 4: Best Local Search (incluye Total Time) ----------
        ws_best_local = workbook.add_worksheet("Best Local Search")
        headers_best_local = ["Instance", "Best Solutions", "Local Methods", "Cost", "Total Time"]
        ws_best_local.write_row(0, 0, headers_best_local, header_format)
        ws_best_local.set_column(0, len(headers_best_local) - 1, default_column_width)
        ws_best_local.freeze_panes(1, 0)

        row = 1
        for plant_name, plant_results in self.results.items():
            entries = plant_results["best_initial"]["entries"]
            initial_cost = entries[0]["cost"]  # coste de la mejor inicial (1st)

            local_best_cost = float('inf')
            best_solutions = []
            best_methods = []
            best_total_times = []

            for solution_id, search_results in plant_results["local_search"]["individual"].items():
                # coste mínimo sobre métodos para esta solución_id
                min_cost_here = min(v.get("cost", float('inf')) for v in search_results.values())
                for method, metrics in search_results.items():
                    current_cost = metrics.get("cost", float('inf'))
                    if current_cost != min_cost_here:
                        continue
                    total_time = metrics.get("total_time", 0.0)
                    if current_cost < local_best_cost:
                        local_best_cost = current_cost
                        best_solutions = [solution_id]
                        best_methods = [LS_ABBR.get(method, method)]
                        best_total_times = [total_time]
                    elif current_cost == local_best_cost:
                        best_solutions.append(solution_id)
                        best_methods.append(LS_ABBR.get(method, method))
                        best_total_times.append(total_time)

            if local_best_cost >= initial_cost:
                best_constructor, _ = next(iter(plant_results["local_search"]["individual"].items()))
                ws_best_local.write_row(
                    row, 0,
                    [plant_name, self.get_alias(best_constructor), "No local search improves the solution", initial_cost, "—"],
                    cell_format,
                )
            else:
                best_solutions_aliases = [self.get_alias(sol) for sol in best_solutions]
                ws_best_local.write_row(
                    row, 0,
                    [plant_name, ", ".join(best_solutions_aliases), ", ".join(best_methods),
                     local_best_cost, ", ".join(f"{t:.4f}s" for t in best_total_times)],
                    cell_format,
                )
            row += 1

        # ---------- Hoja 5: Extended Searches (con Total Time) ----------
        ws_extended = workbook.add_worksheet("Extended Searches")
        headers_extended = ["Instance", "Solution", "Combination", "Cost", "Time", "Total Time"]
        ws_extended.write_row(0, 0, headers_extended, header_format)
        ws_extended.set_column(0, len(headers_extended) - 1, default_column_width)
        ws_extended.freeze_panes(1, 0)

        r = 1
        for plant_name, plant_results in self.results.items():
            for solution_id, extended_results in plant_results["local_search"]["extended"].items():
                solution_id_alias = self.get_alias(solution_id)
                for combination, metrics in extended_results.items():
                    ws_extended.write_row(
                        r, 0,
                        [
                            plant_name,
                            solution_id_alias,
                            combination,
                            metrics.get("cost", "N/A"),
                            metrics.get("time", "N/A"),
                            metrics.get("total_time", "N/A")
                        ],
                        cell_wrap_format
                    )
                    r += 1

        # ---------- Hoja 6: Best Extended Search (incluye Total Time) ----------
        ws_best_ext = workbook.add_worksheet("Best Extended Search")
        headers_best_extended = ["Instance", "Best Solutions", "Combinations", "Cost", "Total Time"]
        ws_best_ext.write_row(0, 0, headers_best_extended, header_format)
        ws_best_ext.set_column(0, len(headers_best_extended) - 1, default_column_width)
        ws_best_ext.freeze_panes(1, 0)

        # Para resumen de combinaciones
        best_count_extended = {}
        best_solution_usage_count = {}
        extended_method_times = {}

        row = 1
        for plant_name, plant_results in self.results.items():
            entries = plant_results["best_initial"]["entries"]
            initial_cost = entries[0]["cost"]
            best_cost_overall = float('inf')
            best_solutions = []
            best_combinations = []
            best_total_times = []

            for solution_id, extended_results in plant_results["local_search"]["extended"].items():
                # coste mínimo sobre combos para esta solución_id
                min_cost_here = min(v.get("cost", float('inf')) for v in extended_results.values())
                for combination, metrics in extended_results.items():
                    current_cost = metrics.get("cost", float('inf'))
                    if current_cost != min_cost_here:
                        continue
                    total_time = metrics.get("total_time", 0.0)
                    if current_cost < best_cost_overall:
                        best_cost_overall = current_cost
                        best_solutions = [solution_id]
                        best_combinations = [combination]
                        best_total_times = [total_time]
                    elif current_cost == best_cost_overall:
                        best_solutions.append(solution_id)
                        best_combinations.append(combination)
                        best_total_times.append(total_time)

            # Acumular tiempos sólo para combos ganadoras (para medias)
            for solution_id, extended_results in plant_results["local_search"]["extended"].items():
                min_cost_here = min(v.get("cost", float('inf')) for v in extended_results.values())
                for combination, metrics in extended_results.items():
                    if metrics.get("cost", float('inf')) == min_cost_here and min_cost_here == best_cost_overall:
                        extended_method_times[combination] = extended_method_times.get(combination, 0.0) + metrics.get("time", 0.0)

            if best_cost_overall >= initial_cost:
                best_constructor, _ = next(iter(plant_results["local_search"]["individual"].items()))
                ws_best_ext.write_row(
                    row, 0,
                    [plant_name, self.get_alias(best_constructor), "No extended search improves the solution", initial_cost, "—"],
                    cell_format,
                )
            else:
                best_solutions_aliases = [self.get_alias(sol) for sol in best_solutions]
                ws_best_ext.write_row(
                    row, 0,
                    [plant_name, ", ".join(best_solutions_aliases), ", ".join(best_combinations),
                     best_cost_overall, ", ".join(f"{t:.4f}s" for t in best_total_times)],
                    cell_format,
                )
                for combination in best_combinations:
                    best_count_extended[combination] = best_count_extended.get(combination, 0) + 1
                for solution in best_solutions:
                    best_solution_usage_count[solution] = best_solution_usage_count.get(solution, 0) + 1
            row += 1

        row += 2
        ws_best_ext.write(row, 0, "Combinations Summary", bold_format); row += 1
        ws_best_ext.write_row(row, 0, ["Combination", "Times Best", "Average Time"], header_format); row += 1
        for combination, count in best_count_extended.items():
            avg_time = extended_method_times.get(combination, 0) / count if count > 0 else 0
            ws_best_ext.write_row(row, 0, [combination, count, avg_time], cell_format)
            row += 1

        row += 2
        ws_best_ext.write(row, 0, "Solutions Summary", bold_format); row += 1
        ws_best_ext.write_row(row, 0, ["Solution", "Times Best"], header_format); row += 1
        for solution, count in best_solution_usage_count.items():
            ws_best_ext.write_row(row, 0, [self.get_alias(solution), count], cell_format)
            row += 1

        # ---------- Hoja 7: Overall Statistics ----------
        ws_summary = workbook.add_worksheet("Overall Statistics")
        headers_summary = ["Algorithm", "Mean (Cost)", "Mean (Time)", "Std. Dev. (rel.)", "Count of Best"]
        ws_summary.write_row(0, 0, headers_summary, header_format)
        ws_summary.set_column(0, len(headers_summary) - 1, default_column_width)
        ws_summary.freeze_panes(1, 0)

        row_idx = 1
        aggregated_constructors = self.aggregate_statistics_across_instances(self.results, "constructors")
        for method_name, method_data in aggregated_constructors.items():
            avg_cost, avg_time, std_dev, total_bests = self.calculate_overall_statistics([method_data])
            alias_name = self.get_alias(method_name)
            ws_summary.write_row(
                row_idx, 0,
                [f"Constructor - {alias_name}", avg_cost, avg_time, std_dev, total_bests],
                cell_format
            )
            row_idx += 1

        workbook.close()

    def calculate_overall_statistics(self, data):
        """
        Compute global statistics (mean cost, mean time, relative std. dev., and #best)
        across all instances.
        """
        all_costs = []
        all_times = []
        all_std_dev = []
        total_bests = 0

        for instance_data in data:
            all_costs.extend(instance_data.get("costs", []))
            all_times.extend(instance_data.get("times", []))
            all_std_dev.extend(instance_data.get("std_devs", []))
            total_bests += instance_data.get("num_bests", 0)

        avg_cost = np.mean(all_costs) if all_costs else 0
        avg_time = np.mean(all_times) if all_times else 0
        std_dev = np.mean(all_std_dev) if all_std_dev else 0

        return avg_cost, avg_time, std_dev, total_bests

    def aggregate_statistics_across_instances(self, results, key):
        """
        Aggregate statistics across instances grouped by method (constructor or local search).
        """
        aggregated = {}

        for plant_results in results.values():
            methods = plant_results.get(key, {})

            if key == "constructors":
                for method_name, method_data in methods.items():
                    if method_name in ["greedy_random_by_row", "random_greedy_by_row", "greedy_random_global",
                                       "greedy_random_row_balanced", "random_greedy_global",
                                       "random_greedy_row_balanced", "global_score_ordering_random"]:
                        averages = method_data.get("averages", {})
                        if not averages:
                            continue
                        std_devs = method_data.get("std_devs", {})
                        num_bests = method_data.get("num_bests", {})
                        times = method_data.get("times", [])
                        ordered_alphas = sorted(averages.keys())

                        for alpha, avg_cost in averages.items():
                            if alpha not in ordered_alphas:
                                continue
                            alpha_index = ordered_alphas.index(alpha)
                            avg_time = times[alpha_index] if alpha_index < len(times) else 0
                            std_dev = std_devs.get(alpha, 0)
                            num_best = num_bests.get(alpha, 0)

                            alpha_name = f"{method_name} (alpha {alpha})"
                            if alpha_name not in aggregated:
                                aggregated[alpha_name] = {"costs": [], "times": [], "num_bests": 0, "std_devs": []}

                            aggregated[alpha_name]["costs"].append(avg_cost)
                            aggregated[alpha_name]["times"].append(avg_time)
                            aggregated[alpha_name]["num_bests"] += num_best
                            aggregated[alpha_name]["std_devs"].append(std_dev)
                    else:
                        if method_name not in aggregated:
                            aggregated[method_name] = {"costs": [], "times": [], "num_bests": 0, "std_devs": []}
                        aggregated[method_name]["costs"].append(method_data.get("average", 0))
                        aggregated[method_name]["times"].append(method_data.get("time", 0))
                        aggregated[method_name]["num_bests"] += method_data.get("num_bests", 0)
                        aggregated[method_name]["std_devs"].append(method_data.get("std_devs", 0))

            elif key == "local_search":
                for search_type, search_methods in methods.items():
                    for solution_name, methods in search_methods.items():
                        search_key = f"{search_type} - {solution_name}"
                        if search_key not in aggregated:
                            aggregated[search_key] = {"costs": [], "times": [], "num_bests": 0}
                        for method_name, method_results in methods.items():
                            costs = method_results.get("costs", [])
                            times = method_results.get("times", [])
                            num_bests = method_results.get("num_bests", 0)
                            aggregated[search_key]["costs"].extend(costs)
                            aggregated[search_key]["times"].extend(times)
                            aggregated[search_key]["num_bests"] += num_bests

        aggregated = {k: v for k, v in aggregated.items() if v["costs"]}
        return aggregated
