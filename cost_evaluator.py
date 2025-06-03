import numpy as np


class CostEvaluator:

    def __init__(self, plant):
        self.n = len(plant.facilities)
        self.sizes = np.array(plant.facilities, dtype=float)
        self.matrix = np.array(plant.matrix, dtype=float)
        self.origin_dist = np.empty(self.n, dtype=float)
        # Índices i,j de la parte triangular superior (i < j) para evaluar el caso completo
        self.i, self.j = np.triu_indices(self.n, k=1)
        # Flujos precalculados para pares (i,j) completos
        self.flows_full = self.matrix[self.i, self.j]
        # Array pre-alocado para diferencias de distancias en la evaluación completa
        self.diffs_full = np.empty_like(self.flows_full)

    def evaluate(self, disposition):
        if sum(len(fila) for fila in disposition) == self.n:
            return self._evaluate_full(disposition)
        else:
            return self._evaluate_partial(disposition)

    def _evaluate_full(self, disposition):
        self._calculate_origin_dist(disposition)

        self.diffs_full[:] = np.abs(self.origin_dist[self.i] - self.origin_dist[self.j])

        cost = float(np.dot(self.flows_full, self.diffs_full))
        return cost


    def _evaluate_partial(self, disposition):
        self._calculate_origin_dist(disposition)

        arrays = [np.asarray(row, dtype=int) for row in disposition if row]
        if not arrays:
            return 0.0
        placed = np.concatenate(arrays)

        p = placed.size

        i_rep = np.repeat(placed, p)
        j_tile = np.tile(placed, p)
        mask = i_rep < j_tile
        i = i_rep[mask]
        j = j_tile[mask]

        diffs = np.abs(self.origin_dist[i] - self.origin_dist[j])
        flows = self.matrix[i, j]

        # 5) Devuelve el producto punto de esos pares
        return float(np.dot(flows, diffs))

    def _calculate_origin_dist(self, disposition):
        self.origin_dist.fill(0.0)

        for row in disposition:
            if not row:
                continue
            idx = np.array(row, dtype=int)
            row_sizes = self.sizes[idx]
            starts = np.concatenate(([0], np.cumsum(row_sizes[:-1])))
            self.origin_dist[idx] = starts + row_sizes / 2