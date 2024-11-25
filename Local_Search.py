import Solution as sol
import numpy as np
from itertools import combinations
from copy import deepcopy

def copy_disposition(disposition):
    return [row[:] for row in disposition]

def swap_facilities(disposition, row, idx1, idx2):
    disposition[row][idx1], disposition[row][idx2] = disposition[row][idx2], disposition[row][idx1]

def create_solution(plant, disposition):
    return sol.Solution(plant=plant, disposition=disposition)

def first_move_swap(solution):
    best_solution = solution
    improved = True

    while improved:
        improved = False
        order_rows = np.random.permutation(len(solution.disposition))

        for i in order_rows:
            for j, k in combinations(range(len(solution.disposition[i])), 2):
                disposition_aux = copy_disposition(solution.disposition)
                swap_facilities(disposition_aux, i, j, k)

                new_solution = create_solution(solution.plant, disposition_aux)
                if new_solution < best_solution:
                    best_solution = new_solution
                    improved = True
                    break
            if improved:
                break

    return best_solution

def best_move_swap(solution):
    best_solution = solution
    improved = True

    while improved:
        improved = False
        current_best_solution = deepcopy(best_solution)

        for i in range(len(solution.disposition)):
            for j, k in combinations(range(len(solution.disposition[i])), 2):
                disposition_aux = copy_disposition(best_solution.disposition)
                swap_facilities(disposition_aux, i, j, k)

                new_solution = create_solution(solution.plant, disposition_aux)
                if new_solution < current_best_solution:
                    current_best_solution = new_solution

        if current_best_solution.cost < best_solution.cost:
            best_solution = current_best_solution
            improved = True

    return best_solution


def first_move(solution):
    best_solution = solution
    improved = True

    while improved:
        improved = False
        order_rows = np.random.permutation(len(solution.disposition))

        for i in order_rows:
            row = solution.disposition[i][:]
            for j in range(len(row)):
                facility = row[j]
                row_copy = row[:j] + row[j+1:]

                for k in range(len(row_copy) + 1):
                    new_row = row_copy[:k] + [facility] + row_copy[k:]

                    disposition_aux = copy_disposition(solution.disposition)
                    disposition_aux[i] = new_row

                    new_solution = create_solution(solution.plant, disposition_aux)
                    if new_solution < best_solution:
                        best_solution = new_solution
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return best_solution

def best_move(solution):
    best_solution = solution
    improved = True

    while improved:
        improved = False
        current_best_solution = best_solution
        best_disposition = None

        for i in range(len(solution.disposition)):
            row = solution.disposition[i][:]
            row_length = len(row)

            for j in range(row_length):
                facility = row[j]
                row_copy = row[:j] + row[j+1:]

                for k in range(row_length):
                    new_row = row_copy[:k] + [facility] + row_copy[k:]
                    disposition_aux = copy_disposition(solution.disposition)
                    disposition_aux[i] = new_row

                    new_solution = create_solution(solution.plant, disposition_aux)
                    if new_solution < current_best_solution:
                        current_best_solution = new_solution

        if current_best_solution < best_solution:
            best_solution = current_best_solution
            improved = True

    return best_solution
