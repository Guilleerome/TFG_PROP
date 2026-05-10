import numpy as np
from typing import List, Optional


class CostEvaluator:
    """
    Evaluador de coste con cálculo delta incremental.

    Mantiene self._current_partial_cost actualizado en cada push_move/pop_move,
    y cost_if_add() calcula el delta sin mutar el estado.

    Asume:
      - Una facility colocada está exactamente en una posición (row, idx).
      - Las matrices de flujo son simétricas (matrix[i,j] == matrix[j,i]).
        Si no lo son, se usa la suma matrix[i,j] + matrix[j,i] como flujo efectivo
        para parejas no ordenadas.
    """

    def __init__(self, plant):
        self.plant = plant
        self.n: int = plant.number
        self.rows: int = plant.rows
        self.capacities: List[int] = plant.capacities
        self.sizes = np.asarray(plant.facilities, dtype=float)
        self.matrix = np.asarray(plant.matrix, dtype=float)

        # Para evaluate_full vectorizado
        self._i_full, self._j_full = np.triu_indices(self.n, k=1)
        self._flows_full = self.matrix[self._i_full, self._j_full]

        self.reset()

    # ------------------------------------------------------------------ #
    # Estado / utilidades
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self._placed: List[List[int]] = [[] for _ in range(self.rows)]
        self._x = np.zeros(self.n, dtype=float)
        self._placed_mask = np.zeros(self.n, dtype=bool)
        self._placed_count = 0
        self._current_partial_cost: float = 0.0
        # fid -> (row, idx) para localizar rápido una facility colocada
        self._loc: dict[int, tuple[int, int]] = {}

    @staticmethod
    def _centers_for_row(row_facilities: List[int], sizes: np.ndarray) -> np.ndarray:
        if not row_facilities:
            return np.empty(0, dtype=float)
        widths = sizes[row_facilities]
        prefix = np.cumsum(np.r_[0.0, widths[:-1]])
        return prefix + widths / 2.0

    def _recompute_row_centers(self, row: int) -> None:
        row_ids = self._placed[row]
        if not row_ids:
            return
        centers = self._centers_for_row(row_ids, self.sizes)
        for idx, fid in enumerate(row_ids):
            self._x[fid] = centers[idx]
            self._loc[fid] = (row, idx)

    def _placed_ids_array(self) -> np.ndarray:
        """IDs de todas las facilities colocadas, en algún orden."""
        return np.flatnonzero(self._placed_mask)

    # ------------------------------------------------------------------ #
    # Evaluaciones desde cero (no incrementales)
    # ------------------------------------------------------------------ #

    def update_new_disposition(self, disposition: List[List[int]]) -> None:
        self.reset()
        for r, row in enumerate(disposition):
            self._placed[r] = list(row)
            for idx, fid in enumerate(row):
                self._placed_mask[fid] = True
                self._loc[fid] = (r, idx)
            self._recompute_row_centers(r)
        self._placed_count = int(self._placed_mask.sum())
        # Recalcular el coste actual desde cero
        self._current_partial_cost = self._compute_partial_from_scratch()

    def _compute_partial_from_scratch(self) -> float:
        m = self._placed_count
        if m <= 1:
            return 0.0
        ids = self._placed_ids_array()
        xi, xj = np.triu_indices(m, k=1)
        A = ids[xi]
        B = ids[xj]
        diffs = np.abs(self._x[A] - self._x[B])
        flows = self.matrix[A, B]
        return float(np.dot(flows, diffs))

    def evaluate_full_with_disposition(self, disposition: List[List[int]]) -> float:
        self.update_new_disposition(disposition)
        return self.evaluate_full()

    def evaluate_full(self) -> float:
        diffs = np.abs(self._x[self._i_full] - self._x[self._j_full])
        return float(np.dot(self._flows_full, diffs))

    def evaluate_partial(self) -> float:
        """Devuelve el coste parcial cacheado (mantenido incrementalmente)."""
        return self._current_partial_cost

    # ------------------------------------------------------------------ #
    # Delta cost — corazón de la estrategia 2
    # ------------------------------------------------------------------ #

    def _delta_for_add(self, row: int, facility: int, position: int) -> float:
        """
        Calcula cuánto cambiaría el coste parcial si añadiéramos `facility` en
        la posición `position` de la fila `row`. NO modifica el estado.
        """
        width_f = float(self.sizes[facility])
        row_facilities = self._placed[row]

        # Posición x donde se colocaría f tras insertarse en `position`.
        # Centros actuales de la fila r (ANTES de la inserción):
        if position == 0:
            x_f = width_f / 2.0
        else:
            # x_f = (suma de anchuras de las facilities en posición < position) + width_f/2
            # Equivalente a: centro de la facility en position-1 + (su mitad) + width_f/2
            left_fid = row_facilities[position - 1]
            x_f = self._x[left_fid] + self.sizes[left_fid] / 2.0 + width_f / 2.0

        # Conjunto de facilities en la fila r que se desplazarán (+width_f).
        shifted_fids = row_facilities[position:]  # los que están en posición >= position

        # Conjunto de facilities que NO se desplazan (resto de placed).
        # placed_arr = todas las placed actualmente.
        placed_arr = self._placed_ids_array()
        if placed_arr.size == 0:
            return 0.0

        # Construimos máscara de "se desplaza" sobre placed_arr.
        shifted_set = set(shifted_fids)
        shifted_mask = np.fromiter(
            (fid in shifted_set for fid in placed_arr),
            dtype=bool,
            count=placed_arr.size,
        )

        x_old = self._x[placed_arr]  # x actuales

        # ---------- (a) Coste de f con todas las placed ----------
        # f compara contra x_new = x_old + width_f * shifted_mask
        # (los desplazados ya están con su nueva x; los que no, con la suya)
        x_others_new = x_old + width_f * shifted_mask
        flows_f = self.matrix[facility, placed_arr]
        diffs_f = np.abs(x_f - x_others_new)
        cost_f = float(np.dot(flows_f, diffs_f))

        # ---------- (b) Cambio en parejas (a, b) con a desplazada y b no ----------
        # Las parejas dentro de S no cambian (ambas se desplazan igual).
        # Las parejas fuera de S no cambian.
        # Las parejas con un extremo en S y otro fuera cambian:
        #     antiguo = |x_old[a] - x_old[b]|
        #     nuevo   = |x_old[a] + width_f - x_old[b]|
        in_s = placed_arr[shifted_mask]
        out_s = placed_arr[~shifted_mask]

        delta_b = 0.0
        if in_s.size > 0 and out_s.size > 0:
            x_a = self._x[in_s][:, None]            # (|S|, 1)
            x_b = self._x[out_s][None, :]           # (1, |out_s|)
            old_diff = np.abs(x_a - x_b)
            new_diff = np.abs(x_a + width_f - x_b)
            flows_ab = self.matrix[in_s[:, None], out_s[None, :]]
            delta_b = float(np.sum(flows_ab * (new_diff - old_diff)))

        return cost_f + delta_b

    # ------------------------------------------------------------------ #
    # Operaciones de modificación
    # ------------------------------------------------------------------ #

    def push_move(self, row: int, facility: int, position: Optional[int] = None) -> None:
        if position is None:
            position = len(self._placed[row])

        # Calcular delta ANTES de modificar el estado
        delta = self._delta_for_add(row, facility, position)

        # Aplicar el cambio
        self._placed[row].insert(position, facility)
        self._placed_mask[facility] = True
        self._placed_count += 1
        self._recompute_row_centers(row)
        # _recompute_row_centers ya actualiza self._loc para todos en la fila

        self._current_partial_cost += delta

    def pop_move(self, row: int, facility: int, position: Optional[int] = None) -> None:
        if position is None:
            if facility not in self._loc:
                raise ValueError(f"Facility {facility} not placed.")
            r, position = self._loc[facility]
            if r != row:
                raise ValueError(f"Facility {facility} is in row {r}, not {row}.")
        else:
            if (
                position >= len(self._placed[row])
                or self._placed[row][position] != facility
            ):
                raise ValueError("Facility and position do not match in pop_move.")

        # Para calcular el delta de eliminación, lo hacemos así:
        # 1) Quitamos la facility (y actualizamos centros + _loc).
        # 2) Calculamos cuánto costaría reinsertarla en la misma posición.
        # 3) El delta de eliminación es -ese valor.
        del self._placed[row][position]
        self._placed_mask[facility] = False
        self._placed_count -= 1
        del self._loc[facility]
        # Actualizar centros (los de la derecha de `position` se desplazan -width_f)
        self._recompute_row_centers(row)

        # Ahora calculamos cuánto costaría reañadirla, eso es lo que debemos restar
        delta_re_add = self._delta_for_add(row, facility, position)
        self._current_partial_cost -= delta_re_add

    def cost_if_add(self, row: int, facility: int, position: int) -> float:
        """
        Devuelve el coste parcial que tendríamos si insertáramos `facility` en
        (row, position). NO modifica el estado. O(m·|fila|) en el peor caso.
        """
        return self._current_partial_cost + self._delta_for_add(row, facility, position)

    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #

    def evaluate(self, disposition: List[List[int]]) -> float:
        return self.evaluate_full_with_disposition(disposition)
