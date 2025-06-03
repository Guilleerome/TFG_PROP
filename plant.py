from cost_evaluator import CostEvaluator


class Plant:
    def __init__(self, name, number, rows, capacities, facilities, matrix):
        self.name = name
        self.number = number
        self.rows = rows
        self.capacities = capacities
        self.facilities = facilities
        self.matrix = matrix
        self.evaluator = CostEvaluator(self)

    def __str__(self):
        return f"Plant {self.name} with {self.number} facilities and {self.rows} rows"

