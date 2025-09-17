from src.io_instances import instances_reader as ir
from src.constructors import constructor as construct
from src.improvers import local_search as ls
import time
import xlsxwriter
import numpy as np
from typing import Dict, Any, List, Tuple

LS_FUN_SIMPLE = {
    "FM": ls.first_move,
    "BM": ls.best_move,
    "FMS": ls.first_move_swap,
    "BMS": ls.best_move_swap,
}

LS_COMBOS = [
    ("FM", "FMS"), ("FM", "BMS"),
    ("BM", "FMS"), ("BM", "BMS"),
    ("FMS", "FM"), ("FMS", "BM"),
    ("BMS", "FM"), ("BMS", "BM"),
]

CONSTRUCTOR_SET = [
    ("Greedy Random Global (α=1.0)", lambda plant: construct.constructor_greedy_random_global(plant, 1.0)),
    ("Greedy Random Global (α=0.75)", lambda plant: construct.constructor_greedy_random_global(plant, 0.75)),
    ("Greedy Random by Row (α=0.75)", lambda plant: construct.constructor_greedy_random_by_row(plant, 0.75)),
]

class MetricsTop3ConstructorsLocalSearch:
    def __init__(
        self,
        excel_path: str = "experiments_top3.xlsx",
        plants=None,
        iterations: int = 30,
    ):
        if not excel_path.endswith(".xlsx"):
            excel_path += ".xlsx"
        self.excel_path = "../results/" + excel_path
        self.iterations = iterations
        self.plants = plants if plants is not None else ir.read_instances("small")
        self.results: Dict[str, Dict[str, Dict[str, Any]]] = {}

        self.run()
        self.write_excel()

    def run(self):
        for plant in self.plants:
            print(f"Processing instance: {plant.name}")
            inst_name = plant.name
            self.results[inst_name] = {}

            for ctor_label, ctor_fun in CONSTRUCTOR_SET:
                self.results[inst_name][ctor_label] = {}
                per_method_runs: Dict[str, List[Tuple[float, float]]] = {}

                for _ in range(self.iterations):
                    t0 = time.time()
                    s0 = ctor_fun(plant)
                    ctor_time = time.time() - t0

                    # Simple LS
                    for m_lbl, m_fun in LS_FUN_SIMPLE.items():
                        t1 = time.time()
                        s1 = m_fun(s0)
                        ls_time = time.time() - t1
                        total_time = ctor_time + ls_time
                        per_method_runs.setdefault(m_lbl, []).append((s1.cost, total_time))

                    # Extended combos
                    for a, b in LS_COMBOS:
                        m1 = LS_FUN_SIMPLE[a]
                        m2 = LS_FUN_SIMPLE[b]
                        t1 = time.time()
                        inter = m1(s0)
                        final = m2(inter)
                        ls_time = time.time() - t1
                        total_time = ctor_time + ls_time
                        combo_lbl = f"{a} + {b}"
                        per_method_runs.setdefault(combo_lbl, []).append((final.cost, total_time))

                # Aggregate metrics per method
                for m_lbl, runs in per_method_runs.items():
                    costs = [c for (c, _) in runs]
                    times_total = [t for (_, t) in runs]
                    best_cost = float(np.min(costs))
                    avg_cost = float(np.mean(costs))
                    total_time_avg = float(np.mean(times_total))
                    self.results[inst_name][ctor_label][m_lbl] = {
                        "best": best_cost,
                        "avg": avg_cost,
                        "total_time_avg": total_time_avg,  # NEW
                        "runs": runs,
                    }

    def write_excel(self):
        wb = xlsxwriter.Workbook(self.excel_path, {"in_memory": True})
        header = wb.add_format({"bold": True, "bg_color": "#D7E4BC", "border": 1, "align": "center"})
        cell = wb.add_format({"border": 1})
        center = wb.add_format({"border": 1, "align": "center"})
        bold = wb.add_format({"bold": True})

        # ---------- Sheet 1: Top3C — Local Searches ----------
        ws = wb.add_worksheet("Top3C — Local Searches")

        # Formatos adicionales para resaltar Best y para la línea separadora por instancia
        cell_center = wb.add_format({"border": 1, "align": "center"})
        best_highlight = wb.add_format({"border": 1, "align": "center", "bold": True, "bg_color": "#FFD700"})
        sep_center = wb.add_format({"border": 1, "align": "center", "bottom": 5})  # borde inferior grueso
        best_highlight_sep = wb.add_format({"border": 1, "align": "center", "bold": True,
                                            "bg_color": "#FFD700", "bottom": 5})

        # Encabezados: Instance | Constructor | (para cada metodo: Best / Average / Total Time)
        method_labels = list(LS_FUN_SIMPLE.keys()) + [f"{a} + {b}" for (a, b) in LS_COMBOS]
        cols = ["Instance", "Constructor"]
        for m in method_labels:
            cols += [f"{m} Best", f"{m} Average", f"{m} Total Time"]

        ws.write_row(0, 0, cols, header)
        ws.set_column(0, len(cols) - 1, 18)
        ws.freeze_panes(1, 0)

        r = 1
        for inst_name, by_ctor in self.results.items():
            # número de constructores que imprimiremos para esta instancia (normalmente 3)
            ctor_names = list(by_ctor.keys())
            n_ctors = len(ctor_names)

            for idx, ctor_label in enumerate(ctor_names):
                metrics_by_method = by_ctor[ctor_label]
                is_last_row_of_instance = (idx == n_ctors - 1)

                # Para resaltar: mejor 'Best' entre todos los métodos de ESTE constructor
                # (si hubiera métodos sin datos, los ignoramos)
                best_values = [v["best"] for v in metrics_by_method.values() if v is not None]
                row_best_min = min(best_values) if best_values else None

                # --- Escribimos celda a celda para poder aplicar formatos condicionales ---
                c = 0

                # Primera columna: Instance
                fmt_first = sep_center if is_last_row_of_instance else cell_center
                ws.write(r, c, inst_name, fmt_first)
                c += 1

                # Segunda columna: Constructor
                ws.write(r, c, ctor_label, fmt_first)
                c += 1

                # Bloques de 3 columnas por metodo: (Best / Average / Total Time)
                for m in method_labels:
                    v = metrics_by_method.get(m)

                    if v is None:
                        # Sin datos para ese metodo
                        best_val = "N/A"
                        avg_val = "N/A"
                        time_val = "N/A"
                        fmt_best = sep_center if is_last_row_of_instance else cell_center
                        fmt_rest = fmt_best
                    else:
                        best_val = v["best"]
                        avg_val = v["avg"]
                        time_val = v["total_time_avg"]

                        # ¿Este "Best" es el mejor de la fila (constructor)?
                        if row_best_min is not None and isinstance(best_val, (int, float)) and best_val == row_best_min:
                            fmt_best = best_highlight_sep if is_last_row_of_instance else best_highlight
                        else:
                            fmt_best = sep_center if is_last_row_of_instance else cell_center

                        fmt_rest = sep_center if is_last_row_of_instance else cell_center

                    # Escribir las 3 columnas de este metodo
                    ws.write(r, c, best_val, fmt_best)
                    c += 1
                    ws.write(r, c, avg_val, fmt_rest)
                    c += 1
                    ws.write(r, c, time_val, fmt_rest)
                    c += 1

                r += 1

        # ---------- Sheet 2: Summary — Top Combos ----------
        # Resumen de las 3 mejores combinaciones (Constructor + Local Search) basadas en la cantidad de 'best'
        # NOTA: 'best' aquí significa alcanzar el mejor coste GLOBAL de la instancia entre TODAS las combinaciones (todos los constructores y métodos).

        ws2 = wb.add_worksheet("Summary — Top Combos")
        ws2.write_row(
            0, 0,
            ["Combo (Constructor + LS)", "Best Count", "Avg Best Cost (where best)", "Avg Best Time (where best)"],
            header
        )
        ws2.set_column(0, 3, 35)
        ws2.freeze_panes(1, 0)

        # 1) Contamos 'best' globales por instancia: cálculo del mejor coste entre TODAS las combinaciones
        best_counter = {}  # combo_key -> {"count": int, "best_costs": [float], "best_times": [float]}

        for inst_name, by_ctor in self.results.items():
            # best global de la instancia (sobre todos los constructores y métodos)
            global_best = float("inf")
            for ctor_label, methods_dict in by_ctor.items():
                for m_lbl, v in methods_dict.items():
                    if v["best"] < global_best:
                        global_best = v["best"]

            # sumar +1 a todas las combinaciones que igualen el best global
            for ctor_label, methods_dict in by_ctor.items():
                for m_lbl, v in methods_dict.items():
                    if v["best"] == global_best:
                        combo_key = f"{ctor_label} | {m_lbl}"
                        entry = best_counter.setdefault(combo_key, {"count": 0, "best_costs": [], "best_times": []})
                        entry["count"] += 1
                        entry["best_costs"].append(v["best"])
                        entry["best_times"].append(v["total_time_avg"])

        # 2) Ordenamos: primero por Best Count (desc), luego Avg Best Cost (asc), luego Avg Best Time (asc)
        def _rank_key(item):
            k, v = item
            count = v["count"]
            avg_cost = (sum(v["best_costs"]) / len(v["best_costs"])) if v["best_costs"] else float("inf")
            avg_time = (sum(v["best_times"]) / len(v["best_times"])) if v["best_times"] else float("inf")
            return (-count, avg_cost, avg_time)

        top3 = sorted(best_counter.items(), key=_rank_key)[:3]

        rr = 1
        for k, v in top3:
            avg_cost = (sum(v["best_costs"]) / len(v["best_costs"])) if v["best_costs"] else "N/A"
            avg_time = (sum(v["best_times"]) / len(v["best_times"])) if v["best_times"] else "N/A"
            ws2.write_row(rr, 0, [k, v["count"], avg_cost, avg_time], center)
            rr += 1

        # (Opcional) Ranking completo para más detalle
        rr += 1
        ws2.write(rr, 0, "Full ranking", bold)
        rr += 1
        ws2.write_row(rr, 0, ["Rank", "Combo", "Best Count", "Avg Best Cost", "Avg Best Time"], header)
        rr += 1
        full_rank = sorted(best_counter.items(), key=_rank_key)
        for idx, (k, v) in enumerate(full_rank, start=1):
            avg_cost = (sum(v["best_costs"]) / len(v["best_costs"])) if v["best_costs"] else "N/A"
            avg_time = (sum(v["best_times"]) / len(v["best_times"])) if v["best_times"] else "N/A"
            ws2.write_row(rr, 0, [idx, k, v["count"], avg_cost, avg_time], center)
            rr += 1

        wb.close()
        print(f"[OK] Wrote {self.excel_path}")
