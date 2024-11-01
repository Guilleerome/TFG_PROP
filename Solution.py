import numpy as np
class Solution:
    def __init__(self, plant=None, disposition=None, cost=None):
        self.plant = plant
        self.disposition = disposition
        self.cost = cost if cost is not None else self.evaluate_cost_D(plant, disposition)

    def __lt__(self, other):
        return self.cost < other.cost

    def __str__(self):
        return f'{self.plant.name} {self.cost}'

    def evaluate_cost(self, plant, disposition):
        cost = 0
        for row in range(len(disposition)):
            for facility in disposition[row]:
                plant.facilities[facility]
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

    def changeDisposition(self, d):
        self.disposition = d
        self.cost = self.evaluate_cost(self.plant, d)

    def evaluate_cost_D(self, plant, disposition):
        n = len(plant.facilities)
        origin_dist = [0] * n
        for row in disposition:
            dist_accum = 0
            for elem in row:
                # print(f"elem: {elem}; val = {dist_accum + plant.facilities[elem] / 2}")
                origin_dist[elem] = dist_accum + plant.facilities[elem] / 2
                dist_accum += plant.facilities[elem]
        np_origin_dist = np.array(origin_dist).reshape(-1, 1)


        m = np.squeeze(np.abs(np.subtract.outer(np_origin_dist, np_origin_dist)))
        result = m * plant.matrix
        #self.cost = result.sum() / 2
        cost = np.sum(result) / 2
        return cost

