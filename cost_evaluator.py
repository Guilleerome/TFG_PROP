
import numpy as np


class CostEvaluator:

    def __init__(self, plant):
        self.n = plant.number
        self.plant = plant
        self.sizes = np.array(plant.facilities, dtype=float)
        self.matrix = np.array(plant.matrix, dtype=float)
        self.origin_dist = np.zeros(self.n, dtype=float)
        # Índices i,j de la parte triangular superior (i < j) para evaluar el caso completo
        self.i, self.j = np.triu_indices(self.n, k=1)
        # Flujos precalculados para pares (i,j) completos
        self.flows_full = self.matrix[self.i, self.j]
        # Array pre-alocado para diferencias de distancias en la evaluación completa
        self.diffs_full = np.empty_like(self.flows_full)
        # _placed[r] es la lista (dinámica) de instalaciones ya añadidas a la fila r
        self._placed = [[] for _ in range(self.plant.rows)]
        self._total_placed = 0

    def reset(self):
        self._placed = [[] for _ in range(self.plant.rows)]
        self._total_placed = 0
        # Reiniciamos todas las posiciones a 0
        self.origin_dist.fill(0.0)

    def push_move(self, row, facility):
        # 1) Insertar en la lista interna
        self._placed[row].append(facility)
        self._total_placed += 1

        # 2) Recalcular únicamente las posiciones centrales de todas las instalaciones en `row`
        self.recalculate_distances(row)

    def pop_move(self, row, facility):
        # 1) Quitar de la lista interna
        self._placed[row].remove(facility)
        self._total_placed -= 1

        # 2) Si la fila quedó vacía, ya no hay posiciones que calcular; las dejamos en 0.
        if not self._placed[row]:
            # Ponemos a 0 el origin_dist[facility] y de cualquier otro que hubiera (pero es el único).
            self.origin_dist[facility] = 0.0
            return

        # 3) Si quedan instalaciones, recalculamos para todas ellas:
        self.recalculate_distances(row)

    def cost_if_add(self, row, f):
        self.push_move(row, f)
        cost = self.evaluate_partial()
        self.pop_move(row, f)
        return cost

    def recalculate_distances(self, row):
        idx = np.array(self._placed[row], dtype=int)
        row_sizes = self.sizes[idx]
        # starts[k] = suma de tamaños de las anteriores en esta fila
        starts = np.concatenate(([0], np.cumsum(row_sizes[:-1])))
        self.origin_dist[idx] = starts + row_sizes / 2

    def evaluate_partial(self):
        if self._total_placed == 0:
            return 0.0

        # 1) Aplanar todas las listas de filas en un array único de length p
        arrays = [np.array(r, dtype=int) for r in self._placed if r]
        placed = np.concatenate(arrays)  # shape = (p,)

        p = placed.size
        # 2) Generar parejas i<j entre las p instalaciones:
        i_rep = np.repeat(placed, p)  # cada facility repetida p veces
        j_tile = np.tile(placed, p)  # cada facility “tileada” p veces
        mask = i_rep < j_tile  # solo nos quedamos con pares (i,j) con i<j

        i = i_rep[mask]
        j = j_tile[mask]

        # 3) Distancias absolutas entre centros
        diffs = np.abs(self.origin_dist[i] - self.origin_dist[j])
        # 4) Flujos correspondientes
        flows = self.matrix[i, j]

        # 5) Producto punto
        return float(np.dot(flows, diffs))

    def evaluate_full(self):
        # Sólo tenemos que rellenar el vector de diferencias y hacer dot
        self.diffs_full[:] = np.abs(self.origin_dist[self.i] - self.origin_dist[self.j])
        return float(np.dot(self.flows_full, self.diffs_full))

    def evaluate_full_with_disposition(self, disposition):
        self.update_new_disposition(disposition)
        return self.evaluate_full()

    def evaluate_partial_with_disposition(self, disposition):
        self.update_new_disposition(disposition)
        return self.evaluate_partial()

    def update_new_disposition(self, disposition):
        self.reset()
        for row_idx, row_list in enumerate(disposition):
            for facility in row_list:
                self.push_move(row_idx, facility)

    def evaluate(self, disposition):
        if sum(len(fila) for fila in disposition) == self.n:
            return self.evaluate_full_with_disposition(disposition)
        else:
            return self.evaluate_partial_with_disposition(disposition)

    def _traditional_evaluate_cost(self, disposition):
        cost = 0
        for row in range(len(disposition)):
            for facility in disposition[row]:
                for i in range(self.plant.number):
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
                                distance_between_facilities += self.plant.facilities[x]
                                # Agregar el costo calculado al costo total
                            # print(facility, i, distance_between_facilities)
                            cost += self.plant.matrix[facility][i] * (
                                        self.plant.facilities[facility] /
                                        2 + distance_between_facilities + self.plant.facilities[i] / 2)
                        except ValueError:
                            # different row
                            pos_facility = disposition[row].index(facility)

                            for other_row in disposition:
                                if i in other_row:
                                    pos_facility = disposition[row].index(facility)
                                    pos_i = other_row.index(i)
                                    distance_to_facility = sum(
                                        self.plant.facilities[x] for x in disposition[row][:pos_facility]) + \
                                                           self.plant.facilities[facility] / 2
                                    distance_to_i = sum(self.plant.facilities[x] for x in other_row[:pos_i]) + \
                                                    self.plant.facilities[i] / 2
                                    distance_between_facilities = abs(distance_to_facility - distance_to_i)
                                    break
                            # print(facility, i, distance_between_facilities)
                            cost += self.plant.matrix[facility][i] * distance_between_facilities
        return cost