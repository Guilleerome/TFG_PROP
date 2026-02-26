
from typing import Dict, Any
from models.plant import Plant
from models.solution import Solution
from .grasp import run_grasp
from .bvns import run_bvns

ALGORITHMS = {
    'grasp': run_grasp,
    'bvns': run_bvns,  # Futuro
    # 'vns': run_vns,    # Futuro
    # 'ils': run_ils,    # Futuro
}


def run_algorithm(
        algorithm_name: str,
        plant: Plant,
        **params
) -> Solution:
    """
    Ejecuta un algoritmo de optimización con los parámetros dados

    Args:
        algorithm_name: Nombre del algoritmo ('grasp', 'bvns', etc.)
        plant: Instancia del problema a resolver
        **params: Parámetros específicos del algoritmo

    Returns:
        Solution: Solución obtenida por el algoritmo

    Raises:
        ValueError: Si el algoritmo no existe

    Examples:
        >>> # Ejecutar GRASP
        >>> sol = run_algorithm(
        ...     'grasp',
        ...     plant,
        ...     constructor_name='greedy_random_by_row',
        ...     ls_sequence=['best_move_swap', 'first_move'],
        ...     alpha=0.67,
        ...     sample_size=42
        ... )
    """
    if algorithm_name not in ALGORITHMS:
        raise ValueError(
            f"Algoritmo '{algorithm_name}' no existe. "
            f"Disponibles: {list(ALGORITHMS.keys())}"
        )

    algorithm_function = ALGORITHMS[algorithm_name]

    # Ejecutar el algoritmo con los parámetros
    solution = algorithm_function(plant=plant, **params)

    return solution