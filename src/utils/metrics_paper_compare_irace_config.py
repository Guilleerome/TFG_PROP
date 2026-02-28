from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import time, csv
import numpy as np
import xlsxwriter

from src.io_instances import instances_reader as ir
from src.algorithms import run_algorithm

# ============================================================================
# Configuración irace — mejores parámetros encontrados
# ============================================================================
IRACE_CONFIG = {
    "algorithm":    "bvns",
    "constructor":  "random_greedy_by_row",
    "alpha":        0.3482,
    "sample_size":  131,
    "ls_sequence":  ["first_move_swap", "best_move"],
    "ls_sample_size": 480,
}


class MetricsIraceVsSota:
    """
    Tabla comparativa:
      Instance | n₁ | MA (SOTA): F, Time(s) | BVNS (irace): F, Time(s)

    - Se ejecutan `iterations` repeticiones por instancia con la config de irace.
    - F = mínimo coste; T = tiempo mínimo entre las repeticiones que logran ese F.
    - MA(SOTA) se carga desde un CSV externo con columnas: instance, n1, F, T.
    - Con filter_to_sota_only=True se descartan instancias que no estén en el CSV.
    """

    def __init__(
        self,
        excel_path: str = "experiments_irace_vs_sota1.xlsx",
        plants: Optional[List[Any]] = None,
        iterations: int = 3,
        sota_csv_path: Optional[str] = None,
        filter_to_sota_only: bool = True,
    ):
        if not excel_path.endswith(".xlsx"):
            excel_path += ".xlsx"
        self.excel_path = "../results/" + excel_path
        self.iterations = iterations
        self.plants = plants if plants is not None else ir.read_instances("small")

        self.sota: Dict[Tuple[str, int], Dict[str, float]] = {}
        if sota_csv_path:
            self.sota = self._load_sota_csv(sota_csv_path)

        self.filter_to_sota_only = filter_to_sota_only and bool(self.sota)
        self.results: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self.skipped: List[Tuple[str, str]] = []

        self._run()
        self._write_excel()

    # ---------- helpers ----------

    def _instance_base_and_m(self, inst_name: str) -> Tuple[str, int]:
        import re
        parts = inst_name.split("_")
        try:
            if inst_name.startswith("AV_") and len(parts) >= 5 and parts[1] == "25":
                return f"AV25_{parts[2]}", int(parts[-1])
            if inst_name.startswith("P24_") and len(parts) >= 3:
                return f"P24_{parts[1]}", int(parts[-1])
            if inst_name.startswith("AV25_"):
                base = inst_name.split()[0]
                m = self._extract_m_from_tail(inst_name)
                return base, m or 0
            if inst_name.startswith("P24_"):
                base = inst_name.split()[0]
                m = self._extract_m_from_tail(inst_name)
                return base, m or 0
            m = re.match(r"^(p\d+)_", inst_name)
            if m:
                return m.group(1), 0
        except Exception:
            pass
        return inst_name, 0

    @staticmethod
    def _extract_m_from_tail(s: str) -> Optional[int]:
        if "(" in s and "/" in s:
            tail = s[s.find("("):]
            digits = [int(ch) for ch in tail if ch.isdigit()]
            return digits[0] if digits else None
        return None

    def _n1_from_plant_or_name(self, plant, inst_name: str, m_hint: int) -> int:
        if hasattr(plant, "capacities") and plant.capacities:
            try:
                return int(plant.capacities[0])
            except Exception:
                pass
        if inst_name.startswith(("AV_", "AV25_")):
            n = 25
        elif inst_name.startswith("P24_"):
            n = 24
        else:
            n = getattr(plant, "number", None) or len(getattr(plant, "facilities", [])) or 0
        m = m_hint if m_hint > 0 else 2
        return int(n // m) if n and m else 0

    def _load_sota_csv(self, path: str) -> Dict[Tuple[str, int], Dict[str, float]]:
        data: Dict[Tuple[str, int], Dict[str, float]] = {}
        with open(path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["instance"].strip(), int(row["n1"]))
                data[key] = {"F": float(row["F"]), "T": float(row["T"])}
        return data

    def _sota_lookup(self, key: Tuple[str, int]) -> Dict[str, float]:
        return self.sota.get(key, {"F": "—", "T": "—"})

    # ---------- core ----------

    def _run(self):
        for plant in self.plants:
            inst_name = plant.name
            base, m = self._instance_base_and_m(inst_name)
            n1 = self._n1_from_plant_or_name(plant, inst_name, m)
            key = (base, n1)

            if self.filter_to_sota_only and key not in self.sota:
                self.skipped.append((inst_name, f"Not in SOTA CSV as ({base}, n1={n1})"))
                continue

            runs: List[Tuple[float, float]] = []
            for _ in range(self.iterations):
                t0 = time.time()
                solution = run_algorithm(
                    algorithm_name=IRACE_CONFIG["algorithm"],
                    plant=plant,
                    constructor_name=IRACE_CONFIG["constructor"],
                    alpha=IRACE_CONFIG["alpha"],
                    sample_size=IRACE_CONFIG["sample_size"],
                    ls_sequence=IRACE_CONFIG["ls_sequence"],
                    ls_sample_size=IRACE_CONFIG["ls_sample_size"],
                )
                elapsed = time.time() - t0
                runs.append((solution.cost, elapsed))

            costs = [c for c, _ in runs]
            best_cost = float(np.min(costs))
            best_time = min(t for c, t in runs if c == best_cost)

            self.results[key] = {
                "m": m,
                "F": best_cost,
                "T": best_time,
                "runs": runs,
            }

    # ---------- excel ----------

    def _write_excel(self):
        wb = xlsxwriter.Workbook(self.excel_path, {"in_memory": True})

        header = wb.add_format({"bold": True, "bg_color": "#D7E4BC", "border": 1,
                                 "align": "center", "font_name": "Arial"})
        subhdr = wb.add_format({"bold": True, "bg_color": "#E8F0D3", "border": 1,
                                 "align": "center", "font_name": "Arial"})
        center = wb.add_format({"border": 1, "align": "center", "font_name": "Arial"})
        gold   = wb.add_format({"border": 1, "align": "center", "bold": True,
                                 "bg_color": "#FFD700", "font_name": "Arial"})
        num_fmt = wb.add_format({"border": 1, "align": "center", "num_format": "0.00",
                                  "font_name": "Arial"})
        gold_num = wb.add_format({"border": 1, "align": "center", "bold": True,
                                   "bg_color": "#FFD700", "num_format": "0.00",
                                   "font_name": "Arial"})

        # ===== Sheet 1: Comparison =====
        ws = wb.add_worksheet("SOTA vs irace")
        ws.freeze_panes(2, 0)

        col_widths = [18, 6, 14, 12, 22, 12]
        for i, w in enumerate(col_widths):
            ws.set_column(i, i, w)

        # Fila 0 — cabeceras de grupo
        ws.write(0, 0, "Instance", header)
        ws.write(0, 1, "n₁", header)
        ws.merge_range(0, 2, 0, 3, "MA (SOTA)", header)
        ws.merge_range(0, 4, 0, 5, "BVNS — config irace", header)

        # Fila 1 — subcabeceras
        for col in [0, 1]:
            ws.write(1, col, "", subhdr)
        ws.write_row(1, 2, ["F", "Time (s)"], subhdr)
        ws.write_row(1, 4, ["F", "Time (s)"], subhdr)

        EPS = 1e-6
        r = 2
        for (base, n1) in sorted(self.results.keys(), key=lambda k: (k[0], -k[1])):
            data = self.results[(base, n1)]
            sota = self._sota_lookup((base, n1))
            sota_F, sota_T = sota["F"], sota["T"]
            our_F, our_T = data["F"], data["T"]
            m = data["m"]

            def _is_num(x):
                return isinstance(x, (int, float))

            beats = _is_num(our_F) and _is_num(sota_F) and (our_F <= sota_F + EPS)

            ratio = f"n/{m}" if 1 < m < 10 else str(m)
            ws.write(r, 0, base, center)
            ws.write(r, 1, ratio, center)
            ws.write(r, 2, sota_F, num_fmt if _is_num(sota_F) else center)
            ws.write(r, 3, sota_T, num_fmt if _is_num(sota_T) else center)
            ws.write(r, 4, our_F, gold_num if beats else num_fmt)
            ws.write(r, 5, our_T, gold if beats else num_fmt)
            r += 1

        # ===== Sheet 2: Raw Data =====
        ws2 = wb.add_worksheet("Raw Data")
        ws2.freeze_panes(1, 0)
        ws2.set_column(0, 4, 20)
        ws2.write_row(0, 0, ["Instance", "n₁", "Run #", "Cost (F)", "Time (s)"], header)

        rr = 1
        for (base, n1), data in self.results.items():
            for i, (cost, t) in enumerate(data["runs"], start=1):
                ws2.write_row(rr, 0, [base, n1, i, cost, t], center)
                rr += 1

        wb.close()
        print(f"[OK] Wrote {self.excel_path}")