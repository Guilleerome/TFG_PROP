import numpy as np

class Solution:
    def __init__(self, plant=None, disposition=None, cost=None):
        self.plant = plant
        self.disposition = disposition

        if cost is not None:
            self.cost = cost
        else:
            self.cost = plant.evaluator.evaluate(disposition)

    def __lt__(self, other):
        return self.cost < other.cost

    def __str__(self):
        return f'{self.plant.name} {self.cost}'

    def change_disposition(self, new_disposition, cost=None):
        self.disposition = new_disposition
        if cost is not None:
            self.cost = cost
        else:
            self.cost = self.plant.evaluator.evaluate(new_disposition)

