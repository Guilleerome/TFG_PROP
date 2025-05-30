import numpy as np


def evaluate_cost(plant, disposition):
    cost = 0
    for row in range(len(disposition)):
        for facility in disposition[row]:
            for i in range(plant.number):
                if facility != i and facility < i:
                    distance_between_facilities = 0
                    try:
                        pos_facility = disposition[row].index(facility)
                        pos_i = disposition[row].index(i)

                        if pos_facility < pos_i:
                            intermedias = disposition[row][pos_facility + 1:pos_i]
                        else:
                            intermedias = disposition[row][pos_i + 1:pos_facility]
                            # Sumar los valores de plant.facilities para las instalaciones intermedias
                        for x in intermedias:
                            distance_between_facilities += plant.facilities[x]
                            # Agregar el costo calculado al costo total
                        # print(facility, i, distance_between_facilities)
                        cost += plant.matrix[facility][i] * (plant.facilities[facility] / 2 + distance_between_facilities + plant.facilities[i] / 2)
                    except ValueError:
                        #different row
                        pos_facility = disposition[row].index(facility)

                        for other_row in disposition:
                            if i in other_row:
                                pos_facility = disposition[row].index(facility)
                                pos_i = other_row.index(i)
                                distance_to_facility = sum(plant.facilities[x] for x in disposition[row][:pos_facility]) + plant.facilities[facility] / 2
                                distance_to_i = sum(plant.facilities[x] for x in other_row[:pos_i]) + plant.facilities[i] / 2
                                distance_between_facilities = abs(distance_to_facility - distance_to_i)
                                break
                        # print(facility, i, distance_between_facilities)
                        cost += plant.matrix[facility][i] * distance_between_facilities
    return cost


def evaluate_cost_matrix(plant, disposition):
    n = len(plant.facilities)
    matrix = np.array(plant.matrix)
    sizes = np.array(plant.facilities, dtype=float)
    origin_dist = np.empty(n, dtype=float)

    for row in disposition:
        row = np.array(row, dtype=int)
        row_sizes = sizes[row]
        starts = np.concatenate(([np.array([0]), np.cumsum(row_sizes[:-1])]))
        origin_dist[row] = starts + row_sizes / 2

    i, j = np.triu_indices(n, k=1)
    diffs = np.abs(origin_dist[i] - origin_dist[j])


    flows = matrix[i, j]
    cost = np.dot(flows, diffs)
    return cost


class Solution:
    def __init__(self, plant=None, disposition=None, cost=None):
        self.plant = plant
        self.disposition = disposition
        self.cost = (
            cost if cost is not None
            else (
                evaluate_cost_matrix(plant, disposition) if plant.rows <= 2
                else evaluate_cost(plant, disposition)
            ) if np.sum([np.sum(row) for row in disposition]) == plant.number
            else evaluate_cost(plant, disposition)
        )


    def __lt__(self, other):
        return self.cost < other.cost

    def __str__(self):
        return f'{self.plant.name} {self.cost}'

    def change_disposition(self, d):
        self.disposition = d
        self.cost = evaluate_cost(self.plant, d)

