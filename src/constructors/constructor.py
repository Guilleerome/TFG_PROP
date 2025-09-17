import math
import numpy as np
from typing import List, Tuple
from src.models.solution import Solution
from src.models.plant import Plant
from copy import deepcopy
import random

from src.constructors.construct_utils import (
    calculate_value_distances_length, reorganize_list,
    build_facilities_by_row, evaluate_best_insertions_in_row,
    evaluate_best_insertion_candidates, sample_pairs,
    select_candidates_greedy_random, select_random_candidates_random_greedy)

def construct_random(plant: Plant) -> Solution:
    rows = plant.rows
    n = plant.number
    disposition = []

    capacities = plant.capacities
    facilities = build_facilities_by_row(capacities)
    for row in range(rows):
        row_facilities = facilities[row]
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
                        calculate_value_distances_length(plant, list(disposition_aux[i].items()),
                                                         factor_length, factor_distances),
                        key=lambda x: x[1],
                        reverse=order
                    )
                    facilities_sorted = [x[0] for x in facilities_sorted]
                    facilities_sorted = reorganize_list(facilities_sorted)
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

    facilities_by_row = build_facilities_by_row(plant.capacities)

    for row in order_rows:
        initial_facility = random.choice(facilities_by_row[row])
        disposition[row].append(initial_facility)
        evaluator.push_move(row, initial_facility)
        facilities_by_row[row].remove(initial_facility)

    cost = 0
    for row in order_rows:
        available_facilities = facilities_by_row[row]

        for _ in range(plant.capacities[row] - 1):
            q = len(available_facilities)
            if q == 0:
                break
            if q <= sample_size:
                available_facilities_sample = available_facilities[:]
            else:
                available_facilities_sample = random.sample(available_facilities, sample_size)

            candidates = evaluate_best_insertions_in_row(row, available_facilities_sample, disposition, evaluator)
            costs = [c[1] for c in candidates]
            min_cost, max_cost = min(costs), max(costs)

            threshold = min_cost + alfa * (max_cost - min_cost)
            rcl = [(f, c, p) for (f, c, p) in candidates if c <= threshold]

            selected_facility, cost, selected_position = random.choice(rcl)
            disposition[row].insert(selected_position, selected_facility)
            evaluator.push_move(row, selected_facility, position=selected_position)
            available_facilities.remove(selected_facility)

    return Solution(plant=plant, disposition=disposition, cost = cost)

def constructor_greedy_random_global(plant: Plant, alfa: float, sample_size:int=40) -> Solution:
    rows = plant.rows
    evaluator = plant.evaluator
    disposition = [[] for _ in range(rows)]

    evaluator.reset()

    facilities_by_row = build_facilities_by_row(plant.capacities)

    capacities_remaining = plant.capacities[:]
    cost = 0

    for row in range(rows):
        initial_facility = random.choice(facilities_by_row[row])
        disposition[row].append(initial_facility)
        evaluator.push_move(row, initial_facility)
        facilities_by_row[row].remove(initial_facility)
        capacities_remaining[row] -= 1

    while any(capacities_remaining):
        rows_with_facilities_remaining = [r for r in range(rows) if facilities_by_row[r]]
        if not rows_with_facilities_remaining:
            break

        candidates_sample = sample_pairs(facilities_by_row, rows_with_facilities_remaining, sample_size)
        candidates = evaluate_best_insertion_candidates(candidates_sample, disposition, evaluator)

        costs = [c[2] for c in candidates]
        min_cost = min(costs)
        max_cost = max(costs)

        threshold = min_cost + alfa * (max_cost - min_cost)

        rcl = [(r, f, c, p) for (r, f, c, p) in candidates if c <= threshold ]

        selected_facility, selected_row, cost, selected_position = random.choice(rcl)

        disposition[selected_row].insert(selected_position, selected_facility)
        evaluator.push_move(selected_row, selected_facility, position=selected_position)
        facilities_by_row[selected_row].remove(selected_facility)
        capacities_remaining[selected_row] -= 1

    return Solution(plant=plant, disposition=disposition)

def constructor_random_greedy_by_row(plant: Plant, alfa: float, sample_size:int=40) -> Solution:
    rows = plant.rows
    evaluator = plant.evaluator
    disposition: List[List[int]] = [[] for _ in range(rows)]

    evaluator.reset()
    facilities_by_row = build_facilities_by_row(plant.capacities)

    for row in range(rows):
        random_facility_start = random.choice(facilities_by_row[row])
        disposition[row].append(random_facility_start)
        evaluator.push_move(row, random_facility_start)
        facilities_by_row[row].remove(random_facility_start)

    cost = 0
    for row in random.sample(range(rows), rows):
        while facilities_by_row[row]:

            available_facilities = select_random_candidates_random_greedy(facilities_by_row[row], alfa, sample_size)
            candidates = evaluate_best_insertions_in_row(row, available_facilities, disposition, evaluator)

            selected_facility, cost, selected_position = min(candidates, key=lambda x: x[1])
            disposition[row].insert(selected_position, selected_facility)
            evaluator.push_move(row, selected_facility, position=selected_position)
            facilities_by_row[row].remove(selected_facility)

    return Solution(plant=plant, disposition=disposition, cost = cost)

def constructor_random_greedy_global(plant: Plant, alfa: float, sample_size: int = 40) -> Solution:
    rows = plant.rows
    evaluator = plant.evaluator
    disposition = [[] for _ in range(rows)]

    evaluator.reset()
    facilities_by_row = build_facilities_by_row(plant.capacities)

    for row in range(rows):
        if facilities_by_row[row]:
            initial_facility = random.choice(facilities_by_row[row])
            disposition[row].append(initial_facility)
            evaluator.push_move(row, initial_facility)
            facilities_by_row[row].remove(initial_facility)

    remaining = [(r, f) for r in range(rows) for f in facilities_by_row[r]]
    cost = 0

    while remaining:
        q = len(remaining)
        s = max(1, int(alfa * q))
        candidates_sample = random.sample(remaining, min(s, sample_size))
        candidates = evaluate_best_insertion_candidates(candidates_sample, disposition, evaluator)

        # Selección puramente greedy
        selected_facility, selected_row, cost, selected_position = min(candidates, key=lambda x: x[2])

        disposition[selected_row].insert(selected_position, selected_facility)
        evaluator.push_move(selected_row, selected_facility, position=selected_position)
        facilities_by_row[selected_row].remove(selected_facility)
        remaining.remove((selected_row, selected_facility))

    return Solution(plant=plant, disposition=disposition, cost=cost)

def constructor_greedy_random_row_balanced(plant: Plant, alfa: float = 0.3, sample_size: int = 40) -> Solution:
    evaluator = plant.evaluator
    rows = plant.rows
    disposition = [[] for _ in range(rows)]
    facilities_by_row = build_facilities_by_row(plant.capacities)
    evaluator.reset()
    weight_flows = 0.8

    # Paso 1: colocar una facility relevante en cada fila (ponderación de score)
    for r in range(rows):
        scores = [(i, weight_flows * sum(plant.matrix[i]) + (1 - weight_flows) * plant.facilities[i]) for i in facilities_by_row[r]]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = scores[:max(1, int(len(scores) * 0.2))]
        i = random.choice(top_k)[0]
        disposition[r].append(i)
        evaluator.push_move(r, i)
        facilities_by_row[r].remove(i)

    # Paso 2: insertar resto con esquema greedy-random (con RCL)
    while any(facilities_by_row):
        row_scores = [(r, sum(plant.facilities[i] for i in disposition[r])) for r in range(rows) if facilities_by_row[r]]
        r = min(row_scores, key=lambda x: x[1])[0]

        candidates = select_random_candidates_random_greedy(facilities_by_row[r], alfa, sample_size)
        evaluated = evaluate_best_insertions_in_row(r, candidates, disposition, evaluator)

        costs = [c[1] for c in evaluated]
        min_cost, max_cost = min(costs), max(costs)
        threshold = min_cost + alfa * (max_cost - min_cost)

        rcl = [c for c in evaluated if c[1] <= threshold]
        selected_facility, _, best_pos = random.choice(rcl)

        disposition[r].insert(best_pos, selected_facility)
        evaluator.push_move(r, selected_facility, best_pos)
        facilities_by_row[r].remove(selected_facility)

    return Solution(plant, disposition)

def constructor_random_greedy_row_balanced(plant: Plant, alfa: float = 0.3, sample_size: int = 40) -> Solution:
    evaluator = plant.evaluator
    rows = plant.rows
    disposition = [[] for _ in range(rows)]
    facilities_by_row = build_facilities_by_row(plant.capacities)
    evaluator.reset()

    for r in range(rows):
        if facilities_by_row[r]:
            i = random.choice(facilities_by_row[r])
            disposition[r].append(i)
            evaluator.push_move(r, i)
            facilities_by_row[r].remove(i)

    while any(facilities_by_row):
        row_scores = [(r, sum(plant.facilities[i] for i in disposition[r])) for r in range(rows) if facilities_by_row[r]]
        r = min(row_scores, key=lambda x: x[1])[0]

        candidates = select_random_candidates_random_greedy(facilities_by_row[r], alfa, sample_size)
        evaluated = evaluate_best_insertions_in_row(r, candidates, disposition, evaluator)
        selected_facility, _, best_pos = min(evaluated, key=lambda x: x[1])

        disposition[r].insert(best_pos, selected_facility)
        evaluator.push_move(r, selected_facility, best_pos)
        facilities_by_row[r].remove(selected_facility)

    return Solution(plant, disposition)

def constructor_global_score_ordering(plant: Plant, weight_flows: float = 0.4) -> Solution:
    evaluator = plant.evaluator
    rows = plant.rows
    disposition = [[] for _ in range(rows)]
    facilities_by_row = build_facilities_by_row(plant.capacities)

    evaluator.reset()
    for r in range(rows):
        scores = [(f, weight_flows * sum(plant.matrix[f]) + (1 - weight_flows) * plant.facilities[f])
                  for f in facilities_by_row[r]]
        scores.sort(key=lambda x: x[1], reverse=True)

        for f, _ in scores:
            best_pos = 0
            best_cost = float('inf')
            for pos in range(len(disposition[r]) + 1):
                cost = evaluator.cost_if_add(r, f, pos)
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos
            disposition[r].insert(best_pos, f)
            evaluator.push_move(r, f, best_pos)

    return Solution(plant, disposition)

def constructor_global_score_ordering_random(plant: Plant, weight_flows: float = 0.2, alfa: float = 0.3) -> Solution:
    evaluator = plant.evaluator
    rows = plant.rows
    disposition = [[] for _ in range(rows)]
    facilities_by_row = build_facilities_by_row(plant.capacities)

    evaluator.reset()
    for r in range(rows):
        scores = [(f, weight_flows * sum(plant.matrix[f]) + (1 - weight_flows) * plant.facilities[f])
                  for f in facilities_by_row[r]]
        remaining = set(f for f, _ in scores)

        while remaining:
            candidates = [item for item in scores if item[0] in remaining]
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_score = candidates[0][1]
            bottom_score = candidates[-1][1]
            threshold = top_score - alfa * (top_score - bottom_score)
            rcl = [f for f, score in candidates if score >= threshold]
            selected_facility = random.choice(rcl)

            best_pos = 0
            best_cost = float('inf')
            for pos in range(len(disposition[r]) + 1):
                cost = evaluator.cost_if_add(r, selected_facility, pos)
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos

            disposition[r].insert(best_pos, selected_facility)
            evaluator.push_move(r, selected_facility, best_pos)
            remaining.remove(selected_facility)

    return Solution(plant, disposition)

