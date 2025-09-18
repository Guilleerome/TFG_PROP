from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import time, math, csv
import numpy as np
import xlsxwriter
import re

from src.io_instances import instances_reader as ir
from src.constructors import constructor as construct
from src.improvers import local_search as ls

LS_FUN = {"FM": ls.first_move, "BM": ls.best_move, "FMS": ls.first_move_swap, "BMS": ls.best_move_swap}
TOP3_COMBOS = [("BMS", "FM"), ("BMS", "BM"), ("FMS", "BM")]

class MetricsPaperComparisonTop3:
    """
    Tabla estilo paper:
      Instance | n₁ | MA(SOTA): F, Time(s) | C2 (α=0.75) BMS+FM: F, Time(s) | C2 (α=0.75) BMS+BM: F, Time(s) | C2 (α=0.75) FMS+BM: F, Time(s)

    - C2 = Greedy Random by Row (alpha=0.75), 30 repeticiones por instancia/combination
    - F = mínimo coste; T = tiempo total (constructor + LS) mínimo entre las repeticiones que logran ese F
    - MA(SOTA) se busca por clave (instance_base, n₁) usando un CSV externo
    - Con filter_to_sota_only=True, se DESCARTAN instancias cuyo (base, n₁) no esté en el CSV
    """

    def __init__(
        self,
        excel_path: str = "experiments_paper_compare.xlsx",
        plants: Optional[List[Any]] = None,
        iterations: int = 30,
        sota_csv_path: Optional[str] = None,      # CSV con columnas: instance,n1,F,T
        filter_to_sota_only: bool = True,         # << NUEVO: descartar lo que no esté en el CSV SOTA
    ):
        if not excel_path.endswith(".xlsx"):
            excel_path += ".xlsx"
        self.excel_path = "../results/" + excel_path
        self.iterations = iterations
        self.plants = plants if plants is not None else ir.read_instances("small")

        # SOTA indexado por (instance_base, n1)
        self.sota: Dict[Tuple[str, int], Dict[str, float]] = {}
        if sota_csv_path:
            self.sota = self._load_sota_csv(sota_csv_path)

        self.filter_to_sota_only = filter_to_sota_only and bool(self.sota)

        # results[(base, n1)] = {"combos": {...}}
        self.results: Dict[Tuple[str, int], Dict[str, Any]] = {}
        # Lista de descartes para hoja informativa
        self.skipped: List[Tuple[str, str]] = []  # (instance_name, reason)

        self._run()
        self._write_excel()

    # ---------- helpers de nombre y n₁ ----------
    def _instance_base_and_m(self, inst_name: str) -> Tuple[str, int]:
        """
        Map:
          'AV_25_1_2_2' -> ('AV25_1', 2)
          'AV_25_3_2_5' -> ('AV25_3', 5)
          'P24_a_2_3'   -> ('P24_a', 3)
          'AV25_1 (n/3)' -> ('AV25_1', 3)
        """
        parts = inst_name.split("_")
        try:
            if inst_name.startswith("AV_") and len(parts) >= 5 and parts[1] == "25":
                base = f"AV25_{parts[2]}"
                m = int(parts[-1])
                return base, m
            if inst_name.startswith("P24_") and len(parts) >= 3:
                base = f"P24_{parts[1]}"
                m = int(parts[-1])
                return base, m
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
                base = m.group(1)
                return base, m.group(2)
        except Exception:
            pass
        return inst_name, 0

    @staticmethod
    def _extract_m_from_tail(s: str) -> Optional[int]:
        # Busca "n/(\d)" entre paréntesis, ej: "AV25_1 (n/3)"
        if "(" in s and "/" in s:
            tail = s[s.find("("):]
            digits = [int(ch) for ch in tail if ch.isdigit()]
            return digits[0] if digits else None
        return None

    def _n1_from_plant_or_name(self, plant, inst_name: str, m_hint: int) -> int:
        # 1) Si la planta trae capacities, úsala (suele ser n1)
        if hasattr(plant, "capacities") and plant.capacities:
            try:
                return int(plant.capacities[0])
            except Exception:
                pass
        # 2) Fallback por nombre: n // m
        if inst_name.startswith(("AV_", "AV25_")):
            n = 25
        elif inst_name.startswith("P24_"):
            n = 24
        else:
            n = getattr(plant, "number", None) or len(getattr(plant, "facilities", [])) or 0
        m = m_hint if m_hint > 0 else 2
        return int(n // m) if n and m else 0

    # ---------- carga SOTA ----------
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

            # Filtro: si se pide filtrar a SOTA y el par (base, n1) NO está en el CSV, descartar
            if self.filter_to_sota_only and key not in self.sota:
                self.skipped.append((inst_name, f"Not in SOTA CSV as ({base}, n1={n1})"))
                continue

            # Ejecutar 30 repeticiones por combo
            print("Running", inst_name, "as", key)
            per_combo_runs: Dict[str, List[Tuple[float, float]]] = {f"{a} + {b}": [] for (a, b) in TOP3_COMBOS}
            for _ in range(self.iterations):
                print("  Iteration", _ + 1)
                # Constructor C2 (Greedy Random by Row, α=0.75)
                t0 = time.time()
                s0 = construct.constructor_greedy_random_by_row(plant, 0.75)
                print("Constructor cost:", s0.cost)
                ctor_time = time.time() - t0

                for a, b in TOP3_COMBOS:
                    t1 = time.time()
                    inter = LS_FUN[a](s0)
                    print("  After", a, "cost:", inter.cost)
                    final = LS_FUN[b](inter)
                    print("  After", b, "cost:", final.cost)
                    ls_time = time.time() - t1
                    total_time = ctor_time + ls_time
                    per_combo_runs[f"{a} + {b}"].append((final.cost, total_time))

            # Agrega: F mínimo y tiempo total mínimo entre los que logran F
            self.results[key] = {"m": m, "combos": {}}
            for combo, runs in per_combo_runs.items():
                costs = [c for (c, _) in runs]
                best_cost = float(np.min(costs))
                best_time = min(t for (c, t) in runs if c == best_cost)
                self.results[key]["combos"][combo] = {"F": best_cost, "T": best_time, "runs": runs}

    # ---------- excel ----------
    def _write_excel(self):
        wb = xlsxwriter.Workbook(self.excel_path, {"in_memory": True})

        header = wb.add_format({"bold": True, "bg_color": "#D7E4BC", "border": 1, "align": "center"})
        subhdr = wb.add_format({"bold": True, "bg_color": "#E8F0D3", "border": 1, "align": "center"})
        center = wb.add_format({"border": 1, "align": "center"})
        gold = wb.add_format({"border": 1, "align": "center", "bold": True, "bg_color": "#FFD700"})

        # ===== Sheet 1 =====
        ws = wb.add_worksheet("Paper Comparison — Top3")
        ws.freeze_panes(2, 0)

        cols = [
            ("Instance", 18), ("n₁", 6),
            ("MA (SOTA)", 14), ("", 12),
            ("C2 (α=0.75) BMS + FM", 20), ("", 12),
            ("C2 (α=0.75) BMS + BM", 20), ("", 12),
            ("C2 (α=0.75) FMS + BM", 20), ("", 12),
        ]
        for i, (_, w) in enumerate(cols):
            ws.set_column(i, i, w)

        ws.write(0, 0, "Instance", header)
        ws.write(0, 1, "n₁", header)
        ws.merge_range(0, 2, 0, 3, "MA (SOTA)", header)
        ws.merge_range(0, 4, 0, 5, "C2 (α=0.75) BMS + FM", header)
        ws.merge_range(0, 6, 0, 7, "C2 (α=0.75) BMS + BM", header)
        ws.merge_range(0, 8, 0, 9, "C2 (α=0.75) FMS + BM", header)

        ws.write_row(1, 2, ["F", "Time (s)"], subhdr)
        ws.write_row(1, 4, ["F", "Time (s)"], subhdr)
        ws.write_row(1, 6, ["F", "Time (s)"], subhdr)
        ws.write_row(1, 8, ["F", "Time (s)"], subhdr)
        ws.write(1, 0, "", subhdr); ws.write(1, 1, "", subhdr)

        r = 2
        for (base, n1) in sorted(self.results.keys(), key=lambda k: (k[0], -k[1])):
            data = self.results[(base, n1)]
            m = data.get("m", 0)
            combos = data["combos"]

            c1 = combos.get("BMS + FM", {"F": "—", "T": "—"})
            c2 = combos.get("BMS + BM", {"F": "—", "T": "—"})
            c3 = combos.get("FMS + BM", {"F": "—", "T": "—"})

            sota = self._sota_lookup((base, n1))
            sota_F, sota_T = sota["F"], sota["T"]

            EPS = 1e-6  # tolerancia numérica para comparar con SOTA

            def _is_num(x):
                return isinstance(x, (int, float))

            def _ties_or_beats_sota(ours, sota):
                return _is_num(ours) and _is_num(sota) and (ours <= sota + EPS)

            ws.write(r, 0, base, center)
            ratio = f"n/{m}" if 1 < m < 10 else f"{m}"
            ws.write(r, 1, ratio, center)

            ws.write(r, 2, sota_F, center)
            ws.write(r, 3, sota_T, center)

            # C2 BMS + FM
            fmt = gold if _ties_or_beats_sota(c1["F"], sota_F) else center
            ws.write(r, 4, c1["F"], fmt)
            ws.write(r, 5, c1["T"], center)

            # C2 BMS + BM
            fmt = gold if _ties_or_beats_sota(c2["F"], sota_F) else center
            ws.write(r, 6, c2["F"], fmt)
            ws.write(r, 7, c2["T"], center)

            # C2 FMS + BM
            fmt = gold if _ties_or_beats_sota(c3["F"], sota_F) else center
            ws.write(r, 8, c3["F"], fmt)
            ws.write(r, 9, c3["T"], center)

            r += 1

        # ===== Sheet 2: Raw Data =====
        ws2 = wb.add_worksheet("Raw Data")
        ws2.freeze_panes(1, 0)
        ws2.set_column(0, 5, 20)
        ws2.write_row(0, 0, ["Instance", "n₁", "Combo", "Run #", "Cost (F)", "Total Time (s)"], header)

        rr = 1
        for (base, n1), data in self.results.items():
            for combo, dd in data["combos"].items():
                for i, (cost, ttot) in enumerate(dd["runs"], start=1):
                    ws2.write_row(rr, 0, [base, n1, combo, i, cost, ttot], center)
                    rr += 1

        wb.close()
        print(f"[OK] Wrote {self.excel_path}")
