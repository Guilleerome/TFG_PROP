"""
bvns.py - BVNS (Basic Variable Neighborhood Search)

Placeholder para implementación futura de BVNS.
"""

from models.plant import Plant
from models.solution import Solution


def run_bvns(
        plant: Plant,
        k_max: int = 5,
        ls_method: str = 'best_move',
        **kwargs
) -> Solution:
    """
    Ejecuta BVNS (Basic Variable Neighborhood Search)

    NOTA: Esta es una implementación placeholder.
    Será implementada en el futuro.

    Args:
        plant: Instancia del problema
        k_max: Número máximo de vecindarios
        ls_method: Metodo de busqueda local
        **kwargs: Parámetros adicionales

    Returns:
        Solution: Solución mejorada

    Raises:
        NotImplementedError: Aún no implementado
    """
    raise NotImplementedError(
        "BVNS aún no está implementado. "
        "Será añadido en futuras versiones."
    )