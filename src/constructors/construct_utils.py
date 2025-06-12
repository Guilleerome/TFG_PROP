import math
import numpy as np
import random
from typing import List, Tuple
from src.models.plant import Plant

def evaluate_best_insertion_candidates(sample: list[tuple[int, int]], disposition: list[list[int]],
                                       evaluator) -> list[tuple[int, int, float, int]]:

    candidates = []
    for r, f in sample:
        best_pos = 0
        best_cost = float('inf')

        for i in range(len(disposition[r]) + 1):
            curr_cost = evaluator.cost_if_add(r, f, position=i)
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_pos = i

        candidates.append((f, r, best_cost, best_pos))

    return candidates

def evaluate_best_insertions_in_row(row: int, facilities: list[int], disposition: list[list[int]],
                                    evaluator) -> list[tuple[int, float, int]]:
    candidates = []
    for f in facilities:
        best_pos = 0
        best_cost = float('inf')
        for i in range(len(disposition[row]) + 1):
            curr_cost = evaluator.cost_if_add(row, f, position=i)
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_pos = i
        candidates.append((f, best_cost, best_pos))
    return candidates

def build_facilities_by_row(capacities: list[int]) -> list[list[int]]:
    facilities_by_row = []
    index = 0
    for capacity in capacities:
        facilities_by_row.append(list(range(index, index + capacity)))
        index += capacity
    return facilities_by_row

def select_random_candidates(row_facilities: List[int], alfa: float, sample_size:int=40) -> List[int]:
    q = len(row_facilities)
    if q <= sample_size:
        return row_facilities
    num_by_alfa = math.ceil(alfa * q)
    s = min(num_by_alfa, sample_size)
    return random.sample(row_facilities, s)

def calculate_value_distances_length(plant: Plant, facilities: List[Tuple[int, int]], factor_length : float, factor_distances: float) -> List[Tuple[int, float]]:
    return [
        (i, (np.sum(plant.matrix[i]) * factor_distances + n * factor_length))
        for i, n in facilities
    ]

def reorganize_list(l: List[int]) -> List[int]:
    new_list = []
    for i in range(0, len(l), 2):
        new_list.append(l[i])
    for i in range(len(l) - 1 - (len(l) % 2), 0, -2):
        if i == -1:
            break
        new_list.append(l[i])
    return new_list

def sample_pairs(facilities_by_row: List[List[int]], rows: List[int], sample_size: int) -> List[Tuple[int, int]]:
    all_pairs = [(r, f) for r in rows for f in facilities_by_row[r]]
    total = len(all_pairs)
    if total <= sample_size:
        return all_pairs
    return random.sample(all_pairs, sample_size)
