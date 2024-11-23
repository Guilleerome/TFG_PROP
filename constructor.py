import instances_reader as ir
from collections import deque
import Solution as sol
from copy import deepcopy

import random


def construct_random(plant):
    rows = plant.rows
    n = plant.number
    disposition = []

    # We distribute the facilities in the rows and shuffle them
    capacity = n // rows
    for i in range(rows):
        if i == rows - 1:  # If it is the last row, we add the remaining facilities
            row_facilities = list(range(i * capacity, n))
        else:
            row_facilities = list(range(i * capacity, (i + 1) * capacity))


        # We shuffle the facilities in the row
        random.shuffle(row_facilities)
        disposition.append(row_facilities)

    return sol.Solution(plant, disposition)


def construct_greedy(plant):
    rows = plant.rows
    n = plant.number
    capacities = plant.capacities
    disposition = []
    disposition_aux = []
    capacity = n / rows
    values = deepcopy(plant.facilities)

    index = 0
    for j in range(rows):
        disposition_aux.append({i: values[i] for i in range(index, index + capacities[j])})
        index += capacities[j]

    for i in range(rows):
        facilities_sorted = sorted(disposition_aux[i].items(), key=lambda x: x[1], reverse=True)
        facilities_sorted = [x[0] for x in facilities_sorted]
        disposition.append(facilities_sorted)

    #idea, when creating the disposition, put in the middle the facilities with the highest values, taking into account the amount of space multiplied by the sum of the distances
    '''for i in range(rows):
        disposition.append(deque())
    for j in range(rows - 1):
        disposition[j].append()
    disposition[rows-1].append(x for x in range(n - capacity.__ceil__(), n))'''

    return sol.Solution(plant, disposition)


def calculate_value_distances_length(plant, facilities, factor_length, factor_distances):
    facilities_calculated = []
    for i, n in facilities:
        v = plant.matrix[i]
        facilities_calculated.append((i, ((sum(v) * factor_distances) + (n * factor_length))))
    return facilities_calculated

def reorganize_list(lista):

    new_list = []
    for i in range(0, len(lista), 2):
        new_list.append(lista[i])
    for i in range(len(lista) - 1 - ((len(lista) % 2)), 0, -2):
        if i == -1:
            break
        new_list.append(lista[i])
    return new_list

    return resultado

def construct_greedy_2(plant):
    rows = plant.rows
    n = plant.number
    capacities = plant.capacities
    disposition = []
    disposition_aux = []
    capacity = n / rows
    values = deepcopy(plant.facilities)
    bestSolution = sol.Solution(cost=float('inf'))

    index = 0
    for j in range(rows):
        disposition_aux.append({i: values[i] for i in range(index, index + capacities[j])})
        index += capacities[j]

    for order in [False, True]:
        factor_length = 0.1
        while factor_length < 1:
            factor_distances = 0.1
            while factor_distances < 1:
                suma_costos_true = 0
                suma_costos_false = 0
                disposition = []
                for i in range(rows):
                    facilities_sorted = sorted(
                        calculate_value_distances_length(plant, disposition_aux[i].items(), factor_length,
                                                         factor_distances),
                        key=lambda x: x[1],
                        reverse=order
                    )
                    facilities_sorted = [x[0] for x in facilities_sorted]
                    facilities_sorted = reorganize_list(facilities_sorted)
                    disposition.append(facilities_sorted)

                new_solution = sol.Solution(plant, disposition)
                if new_solution < bestSolution:
                    bestSolution = new_solution

                factor_distances += 0.1
            factor_length += 0.1

    return bestSolution
