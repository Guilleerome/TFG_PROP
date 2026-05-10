import numpy as np
from typing import List, Optional


class CostEvaluator:
    def __init__(self, plant):
        self.plant = plant
        self.n: int = plant.number
        self.rows: int = plant.rows
        self.capacities: List[int] = plant.capacities
        self.sizes = np.asarray(plant.facilities, dtype=float)
        self.matrix = np.asarray(plant.matrix, dtype=float)

        self._i_full, self._j_full = np.triu_indices(self.n, k=1)
        self._flows_full = self.matrix[self._i_full, self._j_full]

        self._triu_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        self.reset()

    def _triu(self, m: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._triu_cache.get(m)
        if cached is None:
            cached = np.triu_indices(m, k=1)
            self._triu_cache[m] = cached
        return cached

    @staticmethod
    def _centers_for_row(row_facilities: List[int], sizes: np.ndarray) -> np.ndarray:
        if not row_facilities:
            return np.empty(0, dtype=float)
        widths = sizes[row_facilities]
        prefix = np.cumsum(np.r_[0.0, widths[:-1]])
        return prefix + widths / 2.0

    def _recompute_row_centers(self, row: int) -> None:
        row_ids = self._placed[row]
        centers = self._centers_for_row(row_ids, self.sizes)
        for idx, fid in enumerate(row_ids):
            self._x[fid] = centers[idx]

    def reset(self) -> None:
        self._placed: List[List[int]] = [[] for _ in range(self.rows)]
        self._x = np.zeros(self.n, dtype=float)
        self._placed_mask = np.zeros(self.n, dtype=bool)
        self._placed_count = 0
        self._placed_ids: List[int] = []

    def update_new_disposition(self, disposition: List[List[int]]) -> None:
        self.reset()
        for r, row in enumerate(disposition):
            self._placed[r] = list(row)
            self._recompute_row_centers(r)
            for fid in row:
                self._placed_mask[fid] = True
                self._placed_ids.append(fid)
        self._placed_count = len(self._placed_ids)

    def evaluate_full_with_disposition(self, disposition: List[List[int]]) -> float:
        self.update_new_disposition(disposition)
        return self.evaluate_full()

    def evaluate_full(self) -> float:
        diffs = np.abs(self._x[self._i_full] - self._x[self._j_full])
        return float(np.dot(self._flows_full, diffs))

    def evaluate_partial(self) -> float:
        m = self._placed_count
        if m <= 1:
            return 0.0
        ids = np.asarray(self._placed_ids, dtype=np.intp)
        xi, xj = self._triu(m)
        A = ids[xi]
        B = ids[xj]
        diffs = np.abs(self._x[A] - self._x[B])
        flows = self.matrix[A, B]
        return float(np.dot(flows, diffs))

    def push_move(self, row: int, facility: int, position: Optional[int] = None) -> None:
        if position is None:
            self._placed[row].append(facility)
        else:
            self._placed[row].insert(position, facility)

        self._placed_mask[facility] = True
        self._placed_count += 1
        self._placed_ids.append(facility)
        self._recompute_row_centers(row)

    def pop_move(self, row: int, facility: int, position: Optional[int] = None) -> None:
        if position is not None:
            if self._placed[row][position] != facility:
                raise ValueError("Facility and position do not match in pop_move.")
            del self._placed[row][position]
        else:
            self._placed[row].remove(facility)

        self._placed_mask[facility] = False
        self._placed_count -= 1
        self._placed_ids.remove(facility)
        if self._placed[row]:
            self._recompute_row_centers(row)

    def cost_if_add(self, row: int, facility: int, position: int) -> float:
        self.push_move(row, facility, position)
        cost = self.evaluate_partial()
        self.pop_move(row, facility, position)
        return cost

    def evaluate(self, disposition: List[List[int]]) -> float:
        return self.evaluate_full_with_disposition(disposition)
