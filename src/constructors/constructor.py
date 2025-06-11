import math
import numpy as np
from typing import List, Tuple
from src.models.solution import Solution
from src.models.plant import Plant
from sklearn.cluster import KMeans
from copy import deepcopy
import random


def construct_random(plant: Plant) -> Solution:
    rows = plant.rows
    n = plant.number
    disposition = []

    capacity = n // rows
    for i in range(rows):
        if i == rows - 1:  # If it is the last row, we add the remaining facilities
            row_facilities = list(range(i * capacity, n))
        else:
            row_facilities = list(range(i * capacity, (i + 1) * capacity))

        random.shuffle(row_facilities)
        disposition.append(row_facilities)

    return Solution(plant, disposition)

def construct_greedy(plant: Plant) -> Solution:
    rows = plant.rows
    capacities = plant.capacities
    disposition = []
    disposition_aux = []
    values = deepcopy(plant.facilities)

    index = 0
    for j in range(rows):
        disposition_aux.append({i: values[i] for i in range(index, index + capacities[j])})
        index += capacities[j]

    for i in range(rows):
        facilities_sorted = sorted(disposition_aux[i].items(), key=lambda x: x[1], reverse=True)
        facilities_sorted = [x[0] for x in facilities_sorted]
        disposition.append(facilities_sorted)

    return Solution(plant, disposition)

def construct_guillermo(plant: Plant) -> Solution:
    rows = plant.rows
    capacities = plant.capacities
    disposition_aux = []
    values = deepcopy(plant.facilities)
    best_solution = Solution(plant, disposition = [], cost=float('inf'))

    index = 0
    for j in range(rows):
        disposition_aux.append({i: values[i] for i in range(index, index + capacities[j])})
        index += capacities[j]

    for order in [False, True]:
        factor_length = 0.1
        while factor_length < 1:
            factor_distances = 0.1
            while factor_distances < 1:
                disposition_candidate = []
                for i in range(rows):
                    facilities_sorted = sorted(
                        _calculate_value_distances_length(plant, list(disposition_aux[i].items()),
                                                         factor_length,factor_distances),
                        key=lambda x: x[1],
                        reverse=order
                    )
                    facilities_sorted = [x[0] for x in facilities_sorted]
                    facilities_sorted = _reorganize_list(facilities_sorted)
                    disposition_candidate.append(facilities_sorted)

                cost_candidate = plant.evaluator.evaluate(disposition_candidate)
                if cost_candidate < best_solution.cost:
                    best_solution.change_disposition(disposition_candidate, cost_candidate)

                factor_distances += 0.1
            factor_length += 0.1

    return best_solution

def constructor_greedy_random_by_row(plant: Plant, alfa: float, sample_size:int=40) -> Solution:
    rows = plant.rows
    evaluator = plant.evaluator
    disposition = [[] for _ in range(rows)]

    evaluator.reset()
    order_rows = list(range(rows))
    random.shuffle(order_rows)

    index = 0
    facilities_by_row = []
    for capacitiy in plant.capacities:
        facilities_by_row.append(list(range(index, index + capacitiy)))
        index += capacitiy

    cost = 0
    for row in order_rows:
        available_facilities = facilities_by_row[row]

        for _ in range(plant.capacities[row]):
            q = len(available_facilities)
            if q == 0:
                break
            if q <= sample_size:
                available_facilities_sample = available_facilities[:]
            else:
                available_facilities_sample = random.sample(available_facilities, sample_size)

            candidates = []
            min_cost = float('inf')
            max_cost = float('-inf')
            for f in available_facilities_sample:
                cost = evaluator.cost_if_add(row, f)
                candidates.append((f, cost))
                if cost < min_cost: min_cost = cost
                if cost > max_cost: max_cost = cost

            threshold = min_cost + alfa * (max_cost - min_cost)

            rcl = [(f,c) for (f,c) in candidates if c <= threshold]

            selected_facility = random.choice(rcl)[0]
            disposition[row].append(selected_facility)
            evaluator.push_move(row, selected_facility)
            available_facilities.remove(selected_facility)

    return Solution(plant=plant, disposition=disposition, cost = cost)

def constructor_greedy_random_global(plant: Plant, alfa: float, sample_size:int=40) -> Solution:
    rows = plant.rows
    evaluator = plant.evaluator
    disposition = [[] for _ in range(rows)]

    evaluator.reset()
    facilities_by_row = []
    index = 0
    for capacity in plant.capacities:
        facilities_by_row.append(list(range(index, index + capacity)))
        index += capacity

    capacities_remaining = plant.capacities[:]
    cost = 0

    while any(capacities_remaining):
        rows_with_facilities_remaining = [r for r in range(rows) if facilities_by_row[r]]

        if not rows_with_facilities_remaining:
            break

        candidates_sample = _sample_pairs(facilities_by_row, rows_with_facilities_remaining, sample_size)

        min_cost = float('inf')
        max_cost = float('-inf')
        candidates = []
        for (r, f) in candidates_sample:
            cost = evaluator.cost_if_add(r, f)
            candidates.append((f, r, cost))
            if cost < min_cost: min_cost = cost
            if cost > max_cost: max_cost = cost

        threshold = min_cost + alfa * (max_cost - min_cost)

        rcl = [(r, f, c) for (r, f, c) in candidates if c <= threshold ]

        selected_facility, selected_row, _ = random.choice(rcl)

        disposition[selected_row].append(selected_facility)
        evaluator.push_move(selected_row, selected_facility)

        facilities_by_row[selected_row].remove(selected_facility)
        capacities_remaining[selected_row] -= 1

    return Solution(plant=plant, disposition=disposition, cost = cost)

def constructor_random_greedy(plant: Plant, alfa: float, sample_size:int=40) -> Solution:
    rows = plant.rows
    evaluator = plant.evaluator
    disposition: List[List[int]] = [[] for _ in range(rows)]

    evaluator.reset()
    index = 0
    facilities_by_row = []
    for capacitiy in plant.capacities:
        facilities_by_row.append(list(range(index, index + capacitiy)))
        index += capacitiy

    cost = 0
    for row in random.sample(range(rows), rows):
        while facilities_by_row[row]:

            available_facilities = _select_random_candidates(facilities_by_row[row], alfa, sample_size)
            candidates = []
            for f in available_facilities:
                cost = evaluator.cost_if_add(row, f)
                candidates.append((f, cost))

            selected_candidate, _ = min(candidates, key=lambda x: x[1])

            disposition[row].append(selected_candidate)
            evaluator.push_move(row, selected_candidate)
            facilities_by_row[row].remove(selected_candidate)

    return Solution(plant=plant, disposition=disposition, cost = cost)

def _select_random_candidates(row_facilities: List[int], alfa: float, sample_size:int=40) -> List[int]:
    q = len(row_facilities)
    if q <= sample_size:
        return row_facilities
    num_by_alfa = math.ceil(alfa * q)
    s = min(num_by_alfa, sample_size)
    return random.sample(row_facilities, s)

def _calculate_value_distances_length(plant: Plant, facilities: List[Tuple[int, int]], factor_length : float, factor_distances: float) -> List[Tuple[int, float]]:
    return [
        (i, (np.sum(plant.matrix[i]) * factor_distances + n * factor_length))
        for i, n in facilities
    ]

def _reorganize_list(l: List[int]) -> List[int]:
    new_list = []
    for i in range(0, len(l), 2):
        new_list.append(l[i])
    for i in range(len(l) - 1 - (len(l) % 2), 0, -2):
        if i == -1:
            break
        new_list.append(l[i])
    return new_list

def _sample_pairs(facilities_by_row: List[List[int]], rows: List[int], sample_size: int) -> List[Tuple[int, int]]:
    all_pairs = [(r, f) for r in rows for f in facilities_by_row[r]]
    total = len(all_pairs)
    if total <= sample_size:
        return all_pairs
    return random.sample(all_pairs, sample_size)


