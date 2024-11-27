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


def constructor_grasp(plant, alfa):

    rows = plant.rows
    disposition = [[] for _ in range(rows)]

    order_rows = list(range(rows))
    random.shuffle(order_rows)

    index = 0
    facilities_by_row = []
    for capacitiy in plant.capacities:
        facilities_by_row.append(list(range(index, index + capacitiy)))
        index += capacitiy

    for row in order_rows:
        available_facilities = facilities_by_row[row]

        for _ in range(plant.capacities[row]):
            candidates_cost = []

            for facility in available_facilities:
                disposition_aux = copy_disposition(disposition)
                disposition_aux[row].append(facility)

                new_solution = sol.Solution(plant=plant, disposition=disposition_aux)
                cost = new_solution.cost
                candidates_cost.append((facility, cost))

            candidates_cost.sort(key=lambda x:x[1])

            min_cost = candidates_cost[0][1]
            max_cost = candidates_cost[-1][1]
            threshold = min_cost + alfa * (max_cost - min_cost)

            rcl = [candidate for candidate in candidates_cost if candidate[1] <= threshold]

            selected_facility = random.choice(rcl)[0]

            disposition[row].append(selected_facility)
            available_facilities.remove(selected_facility)

    return sol.Solution(plant=plant, disposition=disposition)

def constructor_grasp_2(plant, alfa):
    import random

    rows = plant.rows
    disposition = [[] for _ in range(rows)]  # Solución inicial vacía

    # Crear listas de instalaciones por fila
    facilities_by_row = []
    index = 0
    for capacity in plant.capacities:
        facilities_by_row.append(list(range(index, index + capacity)))
        index += capacity

    # Capacidades restantes por fila
    capacities_remaining = plant.capacities[:]

    while any(capacities_remaining):  # Mientras queden instalaciones por asignar
        candidates_cost = []

        # Para cada fila, probar todas las instalaciones disponibles en esa fila
        for row in range(rows):
            if capacities_remaining[row] > 0:  # Solo considerar filas con capacidad restante
                for facility in facilities_by_row[row]:
                    if facility not in disposition[row]:  # Si la instalación aún no está asignada
                        disposition_aux = copy_disposition(disposition)
                        disposition_aux[row].append(facility)

                        new_solution = sol.Solution(plant=plant, disposition=disposition_aux)
                        cost = new_solution.cost
                        candidates_cost.append((facility, row, cost))  # Guardar (instalación, fila, costo)

        # Ordenar candidatos por costo
        candidates_cost.sort(key=lambda x: x[2], reverse=True)

        # Definir umbral de costos para la lista de candidatos restringida (RCL)
        min_cost = candidates_cost[0][2]
        max_cost = candidates_cost[-1][2]
        threshold = min_cost + alfa * (max_cost - min_cost)

        # Crear la RCL
        rcl = [candidate for candidate in candidates_cost if candidate[2] <= threshold]

        # Seleccionar un candidato al azar de la RCL
        selected_facility, selected_row, _ = random.choice(rcl)

        # Asignar la instalación a la fila correspondiente
        disposition[selected_row].append(selected_facility)
        facilities_by_row[selected_row].remove(selected_facility)
        capacities_remaining[selected_row] -= 1  # Reducir la capacidad restante de la fila

    return sol.Solution(plant=plant, disposition=disposition)

def copy_disposition(disposition):
    return [row[:] for row in disposition]