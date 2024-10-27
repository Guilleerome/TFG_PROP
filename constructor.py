import instances_reader as ir
from collections import deque
import Solution as sol

import random


def construct_random(plant):
    rows = plant.rows
    n = plant.number
    disposition = []
    facilities = list(plant.facilities.keys())

    # We distribute the facilities in the rows and shuffle them
    capacity = n // rows
    for i in range(rows):
        if i == rows - 1:  # If it is the last row, we add the remaining facilities
            row_facilities = facilities[i * capacity:]
        else:
            row_facilities = facilities[i * capacity:(i + 1) * capacity]

        # We shuffle the facilities in the row
        random.shuffle(row_facilities)
        disposition.append(row_facilities)

    return sol.Solution(plant, disposition)


def construct_greedy(plant):
    rows = plant.rows
    n = plant.number
    disposition = []
    disposition_aux = []
    capacity = n / rows
    valores = list(plant.facilities.values())

    for j in range(rows - 1):
        disposition_aux.append(
            {(i + j * rows): valores[(i + j * rows)] for i in range(capacity.__floor__())})
    disposition_aux.append({i: valores[i] for i in range(n - capacity.__ceil__(), n)})

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


def calculate_value_distances_length(plant, facilities):
    factor_length = 0.5
    factor_distances = 1
    facilities_calculated = []
    for i, n in facilities:
        v = plant.matrix[i]
        facilities_calculated.append((i, ((sum(v) * factor_distances) + (n * factor_length))))
    return facilities_calculated

def reorganize_list(lista):
    n = len(lista)
    resultado = []
    for i in range((n + 1) // 2):
        resultado.append(lista[i])
        if i != n - i - 1:
            resultado.append(lista[n - i - 1])
    return resultado
def construct_greedy_2(plant, order):
    rows = plant.rows
    n = plant.number
    disposition = []
    disposition_aux = []
    capacity = n / rows
    valores = list(plant.facilities.values())

    for j in range(rows - 1):
        disposition_aux.append(
            {(i + j * rows): valores[(i + j * rows)] for i in range(capacity.__floor__())})
    disposition_aux.append({i: valores[i] for i in range(n - capacity.__ceil__(), n)})

    for i in range(rows):
        facilities_sorted = sorted(calculate_value_distances_length(plant, disposition_aux[i].items()), key=lambda x: x[1], reverse=order)
        facilities_sorted = [x[0] for x in facilities_sorted]
        facilities_sorted = reorganize_list(facilities_sorted)
        disposition.append(facilities_sorted)

    #idea, when creating the disposition, put in the middle the facilities with the highest values, taking into account the amount of space multiplied by the sum of the distances
    '''for i in range(rows):
        disposition.append(deque())
    for j in range(rows - 1):
        disposition[j].append()
    disposition[rows-1].append(x for x in range(n - capacity.__ceil__(), n))'''

    return sol.Solution(plant, disposition)
