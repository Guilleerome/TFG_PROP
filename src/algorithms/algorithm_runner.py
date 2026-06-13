
from typing import Dict, Any
from src.models.plant import Plant
from src.models.solution import Solution
from .grasp import run_grasp
from .ils import run_ils
from .bvns import run_bvns

ALGORITHMS = {
    'grasp': run_grasp,
    # 'bvns': run_ils,  # Futuro
    'bvns': run_bvns,
    'ils': run_ils,
}


def run_algorithm(
        algorithm_name: str,
        plant: Plant,
        **params
) -> Solution:

    if algorithm_name not in ALGORITHMS:
        raise ValueError(
            f"Algoritmo '{algorithm_name}' no existe. "
            f"Disponibles: {list(ALGORITHMS.keys())}"
        )

    algorithm_function = ALGORITHMS[algorithm_name]

    # Ejecutar el algoritmo con los parámetros
    solution = algorithm_function(plant=plant, **params)

    return solution