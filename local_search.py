import solution as sol
import numpy as np
from itertools import combinations
import random
from copy import deepcopy

def copy_disposition(disposition):
    return [row[:] for row in disposition]

def swap_facilities(disposition, row, idx1, idx2):
    disposition[row][idx1], disposition[row][idx2] = disposition[row][idx2], disposition[row][idx1]

def first_move_swap(solution):
    plant = solution.plant
    evaluator = plant.evaluator

    best_disp = copy_disposition(solution.disposition)
    best_cost = solution.cost
    improved = True

    while improved:
        improved = False
        order_rows = np.random.permutation(len(solution.disposition))

        for i in order_rows:
            for j, k in combinations(range(len(best_disp[i])), 2):
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

    return sol.Solution(plant=plant, disposition=best_disp, cost=best_cost)

def best_move_swap(solution):
    plant = solution.plant
    evaluator = plant.evaluator

    best_disp = copy_disposition(solution.disposition)
    best_cost = solution.cost
    improved = True

    while improved:
        improved = False
        current_best_cost = best_cost
        current_best_disp = None

        for i in range(len(best_disp)):
            for j, k in combinations(range(len(best_disp[i])), 2):
                disposition_aux = copy_disposition(best_disp)
                swap_facilities(disposition_aux, i, j, k)

                cost_aux = evaluator.evaluate(disposition_aux)
                if cost_aux < current_best_cost:
                    current_best_cost = cost_aux
                    current_best_disp = disposition_aux

        if current_best_disp is not None:
            best_disp = current_best_disp
            best_cost = current_best_cost
            improved = True

    return sol.Solution(plant=plant, disposition=best_disp, cost=best_cost)


def first_move(solution):
    plant = solution.plant
    evaluator = plant.evaluator

    best_disp = copy_disposition(solution.disposition)
    best_cost = solution.cost
    improved = True

    while improved:
        improved = False
        order_rows = np.random.permutation(len(solution.disposition))

        for i in order_rows:
            row = best_disp[i]
            for j in range(len(row)):
                facility = row[j]
                row_minus_j = row[:j] + row[j+1:]

                for k in range(len(row_minus_j) + 1):
                    new_row = row_minus_j[:k] + [facility] + row_minus_j[k:]

                    disposition_aux = copy_disposition(best_disp)
                    disposition_aux[i] = new_row

                    cost_aux = evaluator.evaluate(disposition_aux)

                    if cost_aux < best_cost:
                        best_disp = disposition_aux
                        best_cost = cost_aux
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return sol.Solution(plant=plant, disposition=best_disp, cost=best_cost)

def best_move(solution):
    plant = solution.plant
    evaluator = plant.evaluator

    best_disp = copy_disposition(solution.disposition)
    best_cost = solution.cost
    improved = True

    while improved:
        improved = False

        best_change_cost = best_cost
        best_change_disp = None

        for i in range(len(best_disp)):
            row = best_disp[i]
            row_len = len(row)

            for j in range(row_len):
                facility = row[j]
                row_minus_j = row[:j] + row[j+1:]

                for k in range(row_len):
                    new_row = row_minus_j[:k] + [facility] + row_minus_j[k:]
                    disposition_aux = copy_disposition(best_disp)
                    disposition_aux[i] = new_row

                    cost_aux = evaluator.evaluate(disposition_aux)
                    if cost_aux < best_change_cost:
                        best_change_disp = disposition_aux
                        best_change_cost = cost_aux

        if best_change_disp is not None:
            best_disp = best_change_disp
            best_cost = best_change_cost
            improved = True

    return sol.Solution(plant=plant, disposition=best_disp, cost=best_cost)

def combined_local_search(initial_solution):
    current = initial_solution
    improved = True

    while improved:
        improved = False

        bms_sol = best_move_swap(current)
        if bms_sol.cost < current.cost:
            current = bms_sol
            improved = True

            if random.choice([True, False]):
                fm_sol = first_move(current)
                if fm_sol.cost < current.cost:
                    current = fm_sol
            else:
                bm_sol = best_move(current)
                if bm_sol.cost < current.cost:
                    current = bm_sol

    return current

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