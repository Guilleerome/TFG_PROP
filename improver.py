import solution as sol
import numpy as np
import math
import copy

def improve_greedy(solution):
    best_solution = solution
    new_solution = sol.Solution(plant=solution.plant, cost=float('inf'))
    plant = solution.plant
    disposition = solution.disposition
    disposition_aux = copy.deepcopy(disposition)
    used_dispositions = [disposition]
    n = plant.number
    rows = plant.rows
    capacity = n / rows
    rest = n % rows

    for d in range(1, int(math.ceil(capacity.__ceil__() / 2)) + 1):
        origin_dist = [0] * n
        for row in disposition_aux:
            dist_accum = 0
            for elem in row:
                origin_dist[elem] = dist_accum + plant.facilities[elem] / 2
                dist_accum += plant.facilities[elem]

        m = np.subtract.outer(origin_dist, origin_dist)
        matrix_result = m * plant.matrix
        result = np.sum(matrix_result, axis=1)
        result = result / d
        final_result = result + origin_dist

        disposition_aux = []

        for j in range(rows - rest):
            sorted_values = sorted(
                [(i, final_result[i]) for i in range(capacity.__floor__() * j, capacity.__floor__() * (j + 1))],
                key=lambda x: x[1])
            disposition_aux.append([x[0] for x in sorted_values])

        for j in range(rows - rest, rows):
            sorted_values = sorted(
                [(i, final_result[i]) for i in range(n - capacity.__ceil__() * j, n - capacity.__ceil__() * (j - 1))],
                key=lambda x: x[1])
            disposition_aux.append([x[0] for x in sorted_values])

        if used_dispositions.count(disposition_aux) > 2:
            break

        new_solution.change_disposition(disposition_aux)
        if new_solution < best_solution:
            best_solution = copy.deepcopy(new_solution)

        used_dispositions.append(disposition_aux)

    return best_solution




