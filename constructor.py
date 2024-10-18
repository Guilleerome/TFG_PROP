import instances_reader as ir
from collections import deque
import Solution as sol

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
        print(disposition)

    #idea, when creating the disposition, put in the middle the facilities with the highest values, taking into account the amount of space multiplied by the sum of the distances
    '''for i in range(rows):
        disposition.append(deque())
    for j in range(rows - 1):
        disposition[j].append()
    disposition[rows-1].append(x for x in range(n - capacity.__ceil__(), n))'''

    return sol.Solution(plant, disposition)