from typing import Optional
from src.models.solution import Solution
import numpy as np
from itertools import combinations
import random


def copy_disposition(disposition: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in disposition]

def swap_facilities(disposition: list[list[int]], row: int, idx1: int, idx2: int) -> None:
    disposition[row][idx1], disposition[row][idx2] = disposition[row][idx2], disposition[row][idx1]

def _sample_swap_pairs(q: int, s: int) -> list[tuple[int, int]]:
    total_pairs = q * (q - 1) // 2
    if s >= total_pairs:
        return list(combinations(range(q), 2))

    seen = set()
    result = []
    while len(result) < s:
        j = random.randrange(q)
        k = random.randrange(q)
        if j == k:
            continue
        a, b = (j, k) if j < k else (k, j)
        if (a, b) not in seen:
            seen.add((a, b))
            result.append((a, b))
    return result

def _sample_insertion_moves(q: int, s: int) -> list[tuple[int, int]]:
    total_moves = q * (q - 1)
    if s >= total_moves:
        return [(j, k) for j in range(q) for k in range(q) if j != k]

    seen = set()
    result = []
    while len(result) < s:
        j = random.randrange(q)
        k = random.randrange(q)
        if j == k:
            continue
        if (j, k) not in seen:
            seen.add((j, k))
            result.append((j, k))
    return result

def first_move_swap(solution: Solution, s: int=500) -> Solution:
    plant = solution.plant
    evaluator = plant.evaluator

    best_disp = copy_disposition(solution.disposition)
    best_cost = solution.cost
    improved = True

    while improved:
        improved = False
        order_rows = np.random.permutation(len(solution.disposition))

        for i in order_rows:
            q = len(best_disp[i])
            if q < 2:
                continue

            pairs = _sample_swap_pairs(q, s)
            for j, k in pairs:
                disposition_aux = copy_disposition(best_disp)
                swap_facilities(disposition_aux, i, j, k)

                cost_aux = evaluator.evaluate(disposition_aux)

                if cost_aux < best_cost:
                    best_disp = disposition_aux
                    best_cost = cost_aux
                    improved = True
                    break
            if improved:
                break

    return Solution(plant=plant, disposition=best_disp, cost=best_cost)

def best_move_swap(solution: Solution, s: int=500) -> Solution:
    plant = solution.plant
    best_disp = copy_disposition(solution.disposition)
    best_cost = solution.cost

    improved = True

    while improved:
        improved = False

        current_best_disp, current_best_cost = _best_swap(plant, best_disp, best_cost, s)

        if current_best_disp is not None:
            best_disp = current_best_disp
            best_cost = current_best_cost
            improved = True

    return Solution(plant=plant, disposition=best_disp, cost=best_cost)

def first_move(solution: Solution, s: int=500) -> Solution:
    plant = solution.plant
    evaluator = plant.evaluator

    best_disp = copy_disposition(solution.disposition)
    best_cost = solution.cost
    improved = True

    while improved:
        improved = False
        order_rows = np.random.permutation(len(solution.disposition))

        for r in order_rows:
            q = len(best_disp[r])
            if q < 1:
                continue

            pairs = _sample_insertion_moves(q, s)
            for (j,k) in pairs:
                disposition_aux = copy_disposition(best_disp)
                row_aux = disposition_aux[r]

                facility = row_aux[j]
                del row_aux[j]
                row_aux.insert(k, facility)

                cost_aux = evaluator.evaluate(disposition_aux)

                if cost_aux < best_cost:
                    best_disp = disposition_aux
                    best_cost = cost_aux
                    improved = True
                    break
            if improved:
                break

    return Solution(plant=plant, disposition=best_disp, cost=best_cost)

def best_move(solution: Solution, s: int=500) -> Solution:
    plant = solution.plant
    evaluator = plant.evaluator

    best_disp = copy_disposition(solution.disposition)
    best_cost = solution.cost
    improved = True

    while improved:
        improved = False

        best_change_cost = best_cost
        best_change_disp = None

        for r in range(len(best_disp)):
            q = len(best_disp[r])
            if q < 2:
                continue

            pairs = _sample_insertion_moves(q, s)
            for (j,k) in pairs:
                disposition_aux = copy_disposition(best_disp)
                row_aux = disposition_aux[r]
                facility = row_aux[j]
                del row_aux[j]
                row_aux.insert(k, facility)

                cost_aux = evaluator.evaluate(disposition_aux)
                if cost_aux < best_change_cost:
                    best_change_disp = disposition_aux
                    best_change_cost = cost_aux

        if best_change_disp is not None:
            best_disp = best_change_disp
            best_cost = best_change_cost
            improved = True

    return Solution(plant=plant, disposition=best_disp, cost=best_cost)

def combined_local_search(initial_solution: Solution, s: int=500) -> Solution:
    current = initial_solution
    improved = True

    while improved:
        improved = False

        bms_sol = best_move_swap(current, s)
        if bms_sol.cost < current.cost:
            current = bms_sol
            improved = True

            if random.choice([True, False]):
                fm_sol = first_move(current, s)
                if fm_sol.cost < current.cost:
                    current = fm_sol
            else:
                bm_sol = best_move(current)
                if bm_sol.cost < current.cost:
                    current = bm_sol

    return current

def swap_then_first_one_by_one(initial_solution: Solution, s: int=500) -> Solution:
    current = initial_solution
    while True:
        sol_bms, improved = _best_move_swap_once(current, s)
        if not improved:
            return current
        sol_fm = first_move(sol_bms, s)
        current = sol_fm

def _best_move_swap_once(solution: Solution, s: int=500) -> tuple[Solution, bool]:
    plant = solution.plant
    disp = solution.disposition
    current_cost = solution.cost

    best_swap_disp, current_best_cost = _best_swap(plant, disp, current_cost, s)

    if best_swap_disp is None:
        return solution, False
    else:
        new_sol = Solution(plant, best_swap_disp, cost=current_best_cost)
        return new_sol, True

def _best_swap(plant, disp: list[list[int]], current_cost: float, s: int) -> tuple[Optional[list[list[int]]], float]:
    evaluator = plant.evaluator

    best_cost = current_cost
    best_disp = None

    for i in range(len(disp)):
        q = len(disp[i])
        if q < 2:
            continue

        for j, k in _sample_swap_pairs(q, s):
            candidate_disp = copy_disposition(disp)
            swap_facilities(candidate_disp, i, j, k)

            cost_aux = evaluator.evaluate(candidate_disp)
            if cost_aux < best_cost:
                best_cost = cost_aux
                best_disp = candidate_disp

    return best_disp, best_cost

def iterative_local_search(plant, initial_solution):

    current_solution = initial_solution
    improved = True

    while improved:
        improved = False

        fm_swap_solution = first_move_swap(current_solution)
        bm_swap_solution = best_move_swap(current_solution)
        fm_solution = first_move(current_solution)
        bm_solution = best_move(current_solution)

        best_local_solution = min([fm_swap_solution, bm_swap_solution, fm_solution, bm_solution], key=lambda s: s.cost)

        if best_local_solution.cost < current_solution.cost:
            current_solution = best_local_solution
            improved = True

    return current_solution