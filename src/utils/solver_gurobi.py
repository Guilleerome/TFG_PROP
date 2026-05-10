"""
solver_gurobi.py  –  Solver exacto (MIP) para el problema PROP
===============================================================
Basado en la versión funcional original.

Dos modos:
  free_assignment=False (default): orden dentro de cada fila (problema original)
  free_assignment=True:  cualquier facility puede ir a cualquier fila

Sin time limit por defecto → corre hasta gap = 0 (óptimo probado).

Uso:
    python solver_gurobi.py                         # fixed, sin límite
    python solver_gurobi.py --free                  # libre, sin límite
    python solver_gurobi.py --time 3600             # fixed, con límite
    python solver_gurobi.py --free --time 3600      # libre, con límite
"""

import sys
import time
import argparse
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB
import numpy as np


# ──────────────────────────────────────────────
# 1.  Leer instancia
# ──────────────────────────────────────────────

def read_instance(path: str):
    with open(path) as f:
        n = int(f.readline().strip())
        sizes = list(map(int, f.readline().strip().split()))
        capacities = list(map(int, f.readline().strip().split()))
        matrix = []
        for line in f:
            line = line.strip()
            if line:
                matrix.append(list(map(int, line.split())))
    assert len(sizes) == n
    assert sum(capacities) == n
    assert len(matrix) == n and all(len(r) == n for r in matrix)
    return n, sizes, capacities, matrix

def _incumbent_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        obj   = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        gap   = abs(obj - bound) / (abs(obj) + 1e-10) * 100
        t     = model.cbGet(GRB.Callback.RUNTIME)
        print(f"  [t={t:8.1f}s]  Nueva solución: {obj:.2f}  (LB={bound:.2f}, gap={gap:.2f}%)")

# ──────────────────────────────────────────────
# 2.  Solver modo FIXED (asignación fija a filas)
# ──────────────────────────────────────────────

def solve_fixed(path: str, time_limit: float = GRB.INFINITY, mip_gap: float = 0.0,
                verbose: bool = True) -> dict:
    """
    Optimiza solo el ORDEN dentro de cada fila.
    Las facilities 0..C_0-1 van a fila 0, C_0..C_0+C_1-1 a fila 1, etc.
    """
    n, sizes, capacities, matrix = read_instance(path)
    rows = len(capacities)
    flows = np.array(matrix, dtype=float)

    row_of = {}
    fac_in_row = []
    idx = 0
    for r, cap in enumerate(capacities):
        facs = list(range(idx, idx + cap))
        fac_in_row.append(facs)
        for f in facs:
            row_of[f] = r
        idx += cap

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if flows[i, j] + flows[j, i] > 0]

    m = gp.Model("PROP_fixed")
    m.setParam("OutputFlag", 1 if verbose else 0)
    m.setParam("TimeLimit", time_limit)
    m.setParam("MIPGap", mip_gap)
    m.setParam("Presolve", 2)
    m.setParam("Heuristics", 0.5)
    m.setParam("Cuts", 2)
    m.setParam("MIPFocus", 1)
    m.setParam("Symmetry", 2)

    # Variables y[r,k,i], x[i], d[i,j]
    y = {}
    for r in range(rows):
        for k in range(capacities[r]):
            for i in fac_in_row[r]:
                y[r, k, i] = m.addVar(vtype=GRB.BINARY, name=f"y_{r}_{k}_{i}")

    x = {}
    for i in range(n):
        r = row_of[i]
        x[i] = m.addVar(lb=0.0, ub=sum(sizes[j] for j in fac_in_row[r]), name=f"x_{i}")

    d = {}
    for (i, j) in pairs:
        d[i, j] = m.addVar(lb=0.0, name=f"d_{i}_{j}")

    # pos[i] y left[i,j]
    pos = {}
    for i in range(n):
        r = row_of[i]
        pos[i] = m.addVar(lb=0, ub=capacities[r] - 1, vtype=GRB.INTEGER, name=f"pos_{i}")

    left = {}
    for r in range(rows):
        for ii, i in enumerate(fac_in_row[r]):
            for j in fac_in_row[r][ii + 1:]:
                left[j, i] = m.addVar(vtype=GRB.BINARY, name=f"left_{j}_{i}")
                left[i, j] = m.addVar(vtype=GRB.BINARY, name=f"left_{i}_{j}")

    m.update()

    # (1) Cada facility ocupa exactamente una posición
    for r in range(rows):
        for i in fac_in_row[r]:
            m.addConstr(gp.quicksum(y[r, k, i] for k in range(capacities[r])) == 1)

    # (2) Cada posición tiene exactamente una facility
    for r in range(rows):
        for k in range(capacities[r]):
            m.addConstr(gp.quicksum(y[r, k, i] for i in fac_in_row[r]) == 1)

    # (3) pos[i] = Σ_k k*y[r,k,i]
    for r in range(rows):
        for i in fac_in_row[r]:
            m.addConstr(pos[i] == gp.quicksum(k * y[r, k, i] for k in range(capacities[r])))

    # (4) left[j,i] + left[i,j] = 1
    for r in range(rows):
        for ii, i in enumerate(fac_in_row[r]):
            for j in fac_in_row[r][ii + 1:]:
                m.addConstr(left[j, i] + left[i, j] == 1)

    # (5) Ordering via left
    for r in range(rows):
        C = capacities[r]
        for ii, i in enumerate(fac_in_row[r]):
            for j in fac_in_row[r][ii + 1:]:
                m.addConstr(pos[i] - pos[j] >= 1 - C * (1 - left[j, i]))
                m.addConstr(pos[j] - pos[i] >= 1 - C * (1 - left[i, j]))

    # (6) x[i] = w_i/2 + Σ_j w_j * left[j,i]
    for i in range(n):
        r = row_of[i]
        others = [j for j in fac_in_row[r] if j != i]
        m.addConstr(x[i] == sizes[i] / 2.0 + gp.quicksum(sizes[j] * left[j, i] for j in others))

    # (7) d[i,j] >= |x[i] - x[j]|
    for (i, j) in pairs:
        m.addConstr(d[i, j] >= x[i] - x[j])
        m.addConstr(d[i, j] >= x[j] - x[i])

    m.setObjective(
        gp.quicksum((flows[i, j] + flows[j, i]) * d[i, j] for (i, j) in pairs),
        GRB.MINIMIZE
    )

    t0 = time.time()
    m.optimize(_incumbent_callback)
    elapsed = time.time() - t0

    return _extract_result(m, rows, fac_in_row, capacities, pos, x, elapsed)


# ──────────────────────────────────────────────
# 3.  Solver modo FREE (intercambio entre filas)
# ──────────────────────────────────────────────

def solve_free(path: str, time_limit: float = GRB.INFINITY, mip_gap: float = 0.0,
               verbose: bool = True) -> dict:
    """
    Optimiza tanto la ASIGNACIÓN de facilities a filas como el ORDEN dentro de cada fila.
    Solo se respeta que cada fila tenga exactamente capacities[r] facilities.
    """
    n, sizes, capacities, matrix = read_instance(path)
    rows = len(capacities)
    flows = np.array(matrix, dtype=float)

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if flows[i, j] + flows[j, i] > 0]

    m = gp.Model("PROP_free")
    m.setParam("OutputFlag", 1 if verbose else 0)
    m.setParam("TimeLimit", time_limit)
    m.setParam("MIPGap", mip_gap)
    m.setParam("Presolve", 2)
    m.setParam("Heuristics", 0.5)
    m.setParam("Cuts", 2)
    m.setParam("MIPFocus", 1)
    m.setParam("Symmetry", 2)

    # a[i,r] = 1 si facility i está en la fila r
    a = {}
    for i in range(n):
        for r in range(rows):
            a[i, r] = m.addVar(vtype=GRB.BINARY, name=f"a_{i}_{r}")

    # y[i,r,k] = 1 si facility i ocupa la posición k en la fila r
    y = {}
    for i in range(n):
        for r in range(rows):
            for k in range(capacities[r]):
                y[i, r, k] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}_{r}_{k}")

    # pos[i,r] = posición de i en fila r (0 si no está en esa fila)
    pos = {}
    for i in range(n):
        for r in range(rows):
            pos[i, r] = m.addVar(lb=0, ub=capacities[r] - 1, vtype=GRB.INTEGER, name=f"pos_{i}_{r}")

    # left[i,j] = 1 si i está a la izquierda de j en la misma fila
    # Para cada par (i<j)
    left = {}
    for i in range(n):
        for j in range(i + 1, n):
            left[i, j] = m.addVar(vtype=GRB.BINARY, name=f"left_{i}_{j}")
            left[j, i] = m.addVar(vtype=GRB.BINARY, name=f"left_{j}_{i}")

    # same[i,j,r] = 1 si i y j están ambos en la fila r
    same = {}
    for i in range(n):
        for j in range(i + 1, n):
            for r in range(rows):
                same[i, j, r] = m.addVar(vtype=GRB.BINARY, name=f"same_{i}_{j}_{r}")

    x = {}
    for i in range(n):
        max_x = float(max(sum(sorted(sizes, reverse=True)[:capacities[r]]) for r in range(rows)))
        x[i] = m.addVar(lb=0.0, ub=max_x, name=f"x_{i}")

    d = {}
    for (i, j) in pairs:
        d[i, j] = m.addVar(lb=0.0, name=f"d_{i}_{j}")

    m.update()

    # (A1) Cada facility en exactamente una fila
    for i in range(n):
        m.addConstr(gp.quicksum(a[i, r] for r in range(rows)) == 1)

    # (A2) Cada fila tiene exactamente capacities[r] facilities
    for r in range(rows):
        m.addConstr(gp.quicksum(a[i, r] for i in range(n)) == capacities[r])

    # (A3) y[i,r,k] <= a[i,r]
    for i in range(n):
        for r in range(rows):
            for k in range(capacities[r]):
                m.addConstr(y[i, r, k] <= a[i, r])

    # (A4) Si i está en fila r, ocupa exactamente una posición
    for i in range(n):
        for r in range(rows):
            m.addConstr(gp.quicksum(y[i, r, k] for k in range(capacities[r])) == a[i, r])

    # (A5) Cada posición (r,k) tiene exactamente una facility
    for r in range(rows):
        for k in range(capacities[r]):
            m.addConstr(gp.quicksum(y[i, r, k] for i in range(n)) == 1)

    # (A6) pos[i,r] = Σ_k k*y[i,r,k]
    for i in range(n):
        for r in range(rows):
            m.addConstr(pos[i, r] == gp.quicksum(k * y[i, r, k] for k in range(capacities[r])))

    # (A7) same[i,j,r] = a[i,r] AND a[j,r]
    for i in range(n):
        for j in range(i + 1, n):
            for r in range(rows):
                m.addConstr(same[i, j, r] <= a[i, r])
                m.addConstr(same[i, j, r] <= a[j, r])
                m.addConstr(same[i, j, r] >= a[i, r] + a[j, r] - 1)

    # (A8) left[i,j] + left[j,i] = Σ_r same[i,j,r]  (=1 si misma fila, =0 si no)
    for i in range(n):
        for j in range(i + 1, n):
            same_ij = gp.quicksum(same[i, j, r] for r in range(rows))
            m.addConstr(left[i, j] + left[j, i] == same_ij)

    # (A9) Ordering: si i y j están en la misma fila r y left[i,j]=1 => pos[i,r] < pos[j,r]
    for i in range(n):
        for j in range(i + 1, n):
            for r in range(rows):
                C = capacities[r]
                # left[i,j]=1 AND same[i,j,r]=1 => pos[i,r] < pos[j,r]
                m.addConstr(pos[j, r] - pos[i, r] >= 1 - C * (1 - left[i, j]) - C * (1 - same[i, j, r]))
                # left[j,i]=1 AND same[i,j,r]=1 => pos[j,r] < pos[i,r]
                m.addConstr(pos[i, r] - pos[j, r] >= 1 - C * (1 - left[j, i]) - C * (1 - same[i, j, r]))

    # (A10) x[i] = w_i/2 + Σ_{j≠i} w_j * left[j,i]
    for i in range(n):
        others = [j for j in range(n) if j != i]
        m.addConstr(x[i] == sizes[i] / 2.0 + gp.quicksum(sizes[j] * left[j, i] for j in others))

    # (A11) d[i,j] >= |x[i] - x[j]|
    for (i, j) in pairs:
        m.addConstr(d[i, j] >= x[i] - x[j])
        m.addConstr(d[i, j] >= x[j] - x[i])

    m.setObjective(
        gp.quicksum((flows[i, j] + flows[j, i]) * d[i, j] for (i, j) in pairs),
        GRB.MINIMIZE
    )

    t0 = time.time()
    m.optimize(_incumbent_callback)
    elapsed = time.time() - t0

    # Reconstruir disposición desde a[i,r] e y[i,r,k]
    if m.SolCount > 0:
        fac_in_row_sol = [[] for _ in range(rows)]
        for i in range(n):
            for r in range(rows):
                if a[i, r].X > 0.5:
                    fac_in_row_sol[r].append(i)

        disposition = []
        for r in range(rows):
            facs = fac_in_row_sol[r]
            order = sorted(facs, key=lambda i: round(pos[i, r].X))
            disposition.append(order)

        return {
            "obj": m.ObjVal, "bound": m.ObjBound, "gap": m.MIPGap,
            "disposition": disposition, "time": elapsed,
            "status": _status_str(m.Status),
        }

    return {"obj": None, "bound": None, "gap": None,
            "disposition": None, "time": elapsed, "status": _status_str(m.Status)}


# ──────────────────────────────────────────────
# 4.  Helpers
# ──────────────────────────────────────────────

def _status_str(status_code):
    return {
        GRB.OPTIMAL:     "OPTIMAL",
        GRB.TIME_LIMIT:  "TIME_LIMIT",
        GRB.INFEASIBLE:  "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.SUBOPTIMAL:  "SUBOPTIMAL",
    }.get(status_code, f"STATUS_{status_code}")


def _extract_result(m, rows, fac_in_row, capacities, pos, x, elapsed):
    status = _status_str(m.Status)
    if m.SolCount > 0:
        disposition = []
        for r in range(rows):
            facs = fac_in_row[r]
            order = sorted(facs, key=lambda i: round(pos[i].X))
            disposition.append(order)
        return {
            "obj": m.ObjVal, "bound": m.ObjBound, "gap": m.MIPGap,
            "disposition": disposition,
            "x_values": {i: x[i].X for i in range(sum(len(f) for f in fac_in_row))},
            "time": elapsed, "status": status,
        }
    return {"obj": None, "bound": None, "gap": None,
            "disposition": None, "x_values": None, "time": elapsed, "status": status}


def print_result(res: dict, instance_name: str = ""):
    print("\n" + "=" * 60)
    print(f"  Instancia : {instance_name}")
    print(f"  Estado    : {res['status']}")
    print(f"  Tiempo    : {res['time']:.2f} s")
    if res["obj"] is not None:
        print(f"  Obj (UB)  : {res['obj']:.6f}")
        print(f"  Bound (LB): {res['bound']:.6f}")
        print(f"  MIP Gap   : {res['gap']*100:.4f} %")
        print(f"\n  Disposición óptima (índices 0-based):")
        for r, row in enumerate(res["disposition"]):
            print(f"    Fila {r}: {row}")
    else:
        print("  No se encontró solución factible.")
    print("=" * 60)


# ──────────────────────────────────────────────
# 5.  CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solver MIP para el problema PROP")
    parser.add_argument("instance", nargs="?",
                        default=r"W:\CosasUni\TFG_PROP\instances\Parallel-Row Benchmarks\P24_b_2_2.txt",
                        help="Ruta al fichero .txt")
    parser.add_argument("--free", action="store_true",
                        help="Permitir intercambio de facilities entre filas")
    parser.add_argument("--time", type=float, default=None,
                        help="Límite de tiempo en segundos (default: sin límite)")
    parser.add_argument("--gap", type=float, default=0.0,
                        help="Gap MIP objetivo (default: 0.0 = óptimo exacto)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    tl = args.time if args.time is not None else GRB.INFINITY

    if args.free:
        res = solve_free(args.instance, time_limit=tl, mip_gap=args.gap, verbose=not args.quiet)
    else:
        res = solve_free(args.instance, time_limit=tl, mip_gap=args.gap, verbose=not args.quiet)

    print_result(res, Path(args.instance).stem)