class Solution:
    def __init__(self, plant, disposition):
        self.plant = plant
        self.disposition = disposition
        self.cost = self.evaluate_cost(plant, disposition)

    def __str__(self):
        return f'{self.plant.name} {self.cost}'

    def evaluate_cost(self, plant, disposition):
        cost = 0
        for row in disposition:
            for facility in row:
                plant.facilities[facility]
                for i in range(plant.number):
                    if facility != i and facility < i:
                        distance_between_facilities = 0
                        try:
                            pos_facility = row.index(facility)
                            pos_i = row.index(i)

                            if pos_facility < pos_i:
                                intermedias = row[pos_facility + 1:pos_i]
                            else:
                                intermedias = row[pos_i + 1:pos_facility]
                                # Sumar los valores de plant.facilities para las instalaciones intermedias
                            for x in intermedias:
                                distance_between_facilities += plant.facilities[x]
                                # Agregar el costo calculado al costo total
                            cost += plant.matrix[facility][i] * (plant.facilities[facility] / 2 + distance_between_facilities + plant.facilities[i] / 2)
                        except ValueError:
                            #different row
                            pos_facility = row.index(facility)

                            for other_row in disposition:
                                if i in other_row:

                                    pos_facility = row.index(facility)
                                    pos_i = other_row.index(i)
                                    distance_to_facility = sum(plant.facilities[x] for x in row[:pos_facility]) + plant.facilities[facility] / 2
                                    distance_to_i = sum(plant.facilities[x] for x in other_row[:pos_i]) + plant.facilities[i] / 2
                                    distance_between_facilities = abs(distance_to_facility - distance_to_i)
                                    break

                            cost += plant.matrix[facility][i] * distance_between_facilities

        return cost

    def changeDisposition(self, d):
        self.disposition = d
        self.cost = self.evaluate_cost(self.plant, d)