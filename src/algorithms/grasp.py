
from typing import Optional, List, Callable
from src.models.plant import Plant
from src.models.solution import Solution
from .registry import CONSTRUCTORS, LOCAL_SEARCHES


def run_grasp(
        plant: Plant,
        constructor_name: str,
        ls_sequence: Optional[List[str]] = None,
        alpha: float = 0.3,
        sample_size: int = 40,
        ls_sample_size: int = 500,
        n_starts: int = 1,
        **constructor_kwargs
) -> Solution:

    if constructor_name not in CONSTRUCTORS:
        raise ValueError(
            f"Constructor '{constructor_name}' no existe. "
            f"Disponibles: {list(CONSTRUCTORS.keys())}"
        )

    # Validar búsquedas locales
    if ls_sequence:
        for ls_name in ls_sequence:
            if ls_name not in LOCAL_SEARCHES:
                raise ValueError(
                    f"Búsqueda local '{ls_name}' no existe. "
                    f"Disponibles: {list(LOCAL_SEARCHES.keys())}"
                )

    # Validar alpha
    if not (0.1 <= alpha <= 1.0):
        raise ValueError("Alpha debe estar entre 0.1 y 1.0")

    # 1. FASE DE CONSTRUCCIÓN
    constructor = CONSTRUCTORS[constructor_name]

    # Preparar argumentos del constructor
    ctor_args = {'plant': plant}

    # Agregar parámetros según el tipo de constructor
    if constructor_name in ['greedy_random_by_row', 'greedy_random_global',
                            'greedy_random_row_balanced', 'random_greedy_by_row',
                            'random_greedy_global', 'random_greedy_row_balanced']:
        ctor_args['alfa'] = alpha
        ctor_args['sample_size'] = sample_size

    elif constructor_name == 'global_score_ordering_random':
        ctor_args['alfa'] = alpha
        ctor_args['weight_flows'] = constructor_kwargs.get('weight_flows', 0.2)

    elif constructor_name == 'global_score_ordering':
        ctor_args['weight_flows'] = constructor_kwargs.get('weight_flows', 0.2)

    # Agregar cualquier parámetro adicional
    ctor_args.update(constructor_kwargs)

    best = None
    for _ in range(n_starts):
        # Ejecutar constructor
        solution = constructor(**ctor_args)

        # 2. FASE DE BÚSQUEDA LOCAL
        if ls_sequence:
            for ls_name in ls_sequence:
                if ls_name == 'none':
                    continue

                ls_function = LOCAL_SEARCHES[ls_name]
                if ls_function is None:
                    continue

                # Aplicar búsqueda local
                solution = ls_function(solution, s=ls_sample_size)

        if best is None or solution.cost < best.cost:
            best = solution

    return best

