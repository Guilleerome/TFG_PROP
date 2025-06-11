from src.evaluation.cost_evaluator import CostEvaluator


class Plant:
    def __init__(self,
                 name: str,
                 number: int,
                 rows: int,
                 capacities: list,
                 facilities: list,
                 matrix: list[list[int]]) -> None:

        self.name = name
        self.number = number
        self.rows = rows
        self.capacities = capacities
        self.facilities = facilities
        self.matrix = matrix
        self.evaluator = CostEvaluator(self)

    def __str__(self) -> str:
        return f"Plant {self.name} with {self.number} facilities and {self.rows} rows"

