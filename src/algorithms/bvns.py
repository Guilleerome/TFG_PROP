
import random
from typing import List, Optional

from models.plant import Plant
from models.solution import Solution
from .registry import CONSTRUCTORS, LOCAL_SEARCHES
from improvers.local_search import (
    copy_disposition,
    swap_facilities,
    first_move,
    best_move,
    first_move_swap,
    best_move_swap
)


def _perturb_move_k(disposition: list[list[int]], k: int) -> list[list[int]]:
    new_disp = copy_disposition(disposition)

    # Elegir una fila aleatoria con al menos k+1 elementos
    candidates = [r for r, row in enumerate(new_disp) if len(row) > k]
    if not candidates:
        candidates = [r for r, row in enumerate(new_disp) if len(row) >= 2]
    if not candidates:
        return new_disp

    row_idx = random.choice(candidates)
    row = new_disp[row_idx]
    q = len(row)

    # Seleccionar k instalaciones distintas y moverlas a posiciones aleatorias
    indices = random.sample(range(q), min(k, q))
    for idx in sorted(indices, reverse=True):
        facility = row.pop(idx)
        new_pos = random.randrange(len(row) + 1)
        row.insert(new_pos, facility)

    return new_disp


def _perturb_segment_inversion(disposition: list[list[int]]) -> list[list[int]]:
    """
    Perturbación 4: invierte un segmento aleatorio en una fila aleatoria.
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
    row[i:j+1] = row[i:j+1][::-1]

    return new_disp


def _perturb_double_swap(disposition: list[list[int]]) -> list[list[int]]:
    """
    Perturbación 5: en dos filas aleatorias distintas, intercambia 2
    facilities contiguas con otras 2 contiguas dentro de la misma fila.
    """
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
        row[i+1], row[j+1] = row[j+1], row[i+1]

    return new_disp


def _apply_perturbation(disposition: list[list[int]], counter: int) -> list[list[int]]:

    if counter <= 3:
        return _perturb_move_k(disposition, k=counter)
    elif counter == 4:
        return _perturb_segment_inversion(disposition)
    else:
        return _perturb_double_swap(disposition)


def run_bvns(
        plant: Plant,
        constructor_name: str,
        ls_sequence: Optional[List[str]] = None,
        alpha: float = 0.3,
        sample_size: int = 40,
        ls_sample_size: int = 500,
        **constructor_kwargs
) -> Solution:
    """
    Ejecuta BVNS (Basic Variable Neighborhood Search).

    Estructura:
        1. Construir solución inicial con el constructor dado.
        2. Aplicar búsqueda local inicial.
        3. Bucle BVNS:
            a. Perturbar según el contador de no-mejora (1-5).
            b. Aplicar búsqueda local a la solución perturbada.
            c. Si mejora → actualizar mejor solución y reiniciar contador.
               Si no mejora → incrementar contador.
            d. Si contador > 5 → parar.
    """
    from src.algorithms.grasp import run_grasp

    # Paso 1: Construir solución inicial con GRASP (constructor + LS)
    current = run_grasp(
        plant,
        constructor_name=constructor_name,
        ls_sequence=ls_sequence,
        alpha=alpha,
        sample_size=sample_size,
        ls_sample_size=ls_sample_size,
        **constructor_kwargs
    )

    best = current
    evaluator = plant.evaluator

    ls_functions = []
    if ls_sequence:
        ls_functions = [LOCAL_SEARCHES[ls_name] for ls_name in ls_sequence if ls_name in LOCAL_SEARCHES]

    no_improve_count = 1

    while no_improve_count <= 5:
        perturbed_disp = _apply_perturbation(best.disposition, no_improve_count)
        perturbed_cost = evaluator.evaluate(perturbed_disp)
        perturbed = Solution(plant, perturbed_disp, perturbed_cost)

        improved_solution = perturbed
        for ls_func in ls_functions:
            improved_solution = ls_func(improved_solution, ls_sample_size)

        if improved_solution.cost < best.cost:
            best = improved_solution
            no_improve_count = 1
        else:
            no_improve_count += 1

    return best