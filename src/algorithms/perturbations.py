import random

from src.improvers.local_search import copy_disposition, swap_facilities


# ── Perturbaciones básicas ──────────────────────────────────────────────────

def perturb_move_k(disposition: list[list[int]], k: int) -> list[list[int]]:
    new_disp = copy_disposition(disposition)


    for _ in range(k):
        candidates = [r for r, row in enumerate(new_disp) if len(row) > 2]

        if not candidates:
            return new_disp

        weights = [len(new_disp[r]) for r in candidates]
        row_idx = random.choices(candidates, weights=weights, k=1)[0]
        row = new_disp[row_idx]
        idx = random.randrange(len(row))
        facility = row.pop(idx)
        new_pos = random.randrange(len(row) + 1)
        row.insert(new_pos, facility)

    return new_disp

def perturb_segment_inversion(disposition: list[list[int]]) -> list[list[int]]:
    """
    Invierte un segmento aleatorio en una fila aleatoria.
    """
    new_disp = copy_disposition(disposition)

    candidates = [r for r, row in enumerate(new_disp) if len(row) >= 2]
    if not candidates:
        return new_disp

    row_idx = random.choice(candidates)
    row = new_disp[row_idx]
    q = len(row)

    # Elegir dos índices distintos para definir el segmento
    i, j = sorted(random.sample(range(q), 2))
    row[i:j + 1] = row[i:j + 1][::-1]

    return new_disp

def perturb_double_swap(disposition: list[list[int]]) -> list[list[int]]:
    new_disp = copy_disposition(disposition)

    candidates = [r for r, row in enumerate(new_disp) if len(row) >= 4]
    if len(candidates) < 2:
        # Si no hay dos filas con >=4, intentar con >=2
        candidates = [r for r, row in enumerate(new_disp) if len(row) >= 2]
    if len(candidates) < 2:
        return new_disp

    row_a, row_b = random.sample(candidates, 2)

    for row_idx in [row_a, row_b]:
        row = new_disp[row_idx]
        q = len(row)

        # Elegir dos pares de posiciones contiguas distintos
        # Par 1: posición aleatoria i, instalaciones en [i, i+1]
        # Par 2: posición aleatoria j != i, instalaciones en [j, j+1]
        possible_starts = list(range(q - 1))
        if len(possible_starts) < 2:
            # Fila demasiado corta, hacer swap simple
            swap_facilities(new_disp, row_idx, 0, 1)
            continue

        i = random.choice(possible_starts)
        possible_starts_j = [s for s in possible_starts if abs(s - i) >= 2]

        if not possible_starts_j:
            # No hay espacio para dos pares no solapados, swap simple
            j = random.choice([s for s in possible_starts if s != i])
            row[i], row[j] = row[j], row[i]
            continue

        j = random.choice(possible_starts_j)

        # Intercambiar los dos pares completos
        row[i], row[j] = row[j], row[i]
        row[i + 1], row[j + 1] = row[j + 1], row[i + 1]

    return new_disp

def perturb_move_k_multi_row(disposition: list[list[int]], k: int = 2) -> list[list[int]]:
    new_disp = copy_disposition(disposition)
    for _ in range(len(disposition)):
        new_disp = perturb_move_k(new_disp, k)
    return new_disp


# Catálogo completo de perturbaciones disponibles para el post-optimizer
ALL_PERTURBATIONS = {
    'move_k1':         lambda d: perturb_move_k(d, k=1),
    'move_k2':         lambda d: perturb_move_k(d, k=2),
    'move_k3':         lambda d: perturb_move_k(d, k=3),
    'segment_inv':     lambda d: perturb_segment_inversion(d),
    'double_swap':     lambda d: perturb_double_swap(d),
    'multi_row_k2':    lambda d: perturb_move_k_multi_row(d, k=2),
}
