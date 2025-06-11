from src.models.plant import Plant
from typing import Optional

class Solution:
    def __init__(self, plant: Plant, disposition: list[list[int]],
                 cost: Optional[float]=None):

        self.plant = plant
        self.disposition = disposition

        if cost is not None:
            self.cost = cost
        else:
            self.cost = plant.evaluator.evaluate(disposition)

    def __lt__(self, other: 'Solution') -> bool:
        return self.cost < other.cost

    def __str__(self) -> str:
        return f'{self.plant.name} {self.cost}'

    def __repr__(self) -> str:
        return f'Solution(plant={self.plant.name}, cost={self.cost})'

    def change_disposition(self, new_disposition: list[list[int]], cost: Optional[float]=None) -> None:
        self.disposition = new_disposition
        if cost is not None:
            self.cost = cost
        else:
            self.cost = self.plant.evaluator.evaluate(new_disposition)

