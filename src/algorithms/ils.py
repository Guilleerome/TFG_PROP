
import random
from typing import List, Optional

from src.models.plant import Plant
from src.models.solution import Solution
from .registry import CONSTRUCTORS, LOCAL_SEARCHES
from src.improvers.local_search import (
    first_move,
    best_move,
    first_move_swap,
    best_move_swap
)
from .perturbations import (
    perturb_move_k,
    perturb_segment_inversion,
    perturb_double_swap
)


def _apply_perturbation(disposition: list[list[int]], counter: int) -> list[list[int]]:
    if counter <= 3:
        return perturb_move_k(disposition, k=counter)
    elif counter == 4:
        return perturb_segment_inversion(disposition)
    else:
        return perturb_double_swap(disposition)


def run_ils(
        plant: Plant,
        constructor_name: str,
        ls_sequence: Optional[List[str]] = None,
        alpha: float = 0.3,
        sample_size: int = 40,
        ls_sample_size: int = 500,
        n_starts: int = 10,
        **constructor_kwargs
) -> Solution:

    from src.algorithms.grasp import run_grasp

    current = run_grasp(
        plant,
        constructor_name=constructor_name,
        ls_sequence=ls_sequence,
        alpha=alpha,
        sample_size=sample_size,
        ls_sample_size=ls_sample_size,
        n_starts=n_starts,
        **constructor_kwargs
    )

    evaluator = plant.evaluator

    ls_functions = []
    if ls_sequence:
        ls_functions = [LOCAL_SEARCHES[ls_name] for ls_name in ls_sequence if ls_name in LOCAL_SEARCHES]

    no_improve_count = 1

    while no_improve_count <= 5:
        perturbed_disp = _apply_perturbation(current.disposition, no_improve_count)
        perturbed_cost = evaluator.evaluate(perturbed_disp)
        perturbed = Solution(plant, perturbed_disp, perturbed_cost)

        improved_solution = perturbed
        for ls_func in ls_functions:
            improved_solution = ls_func(improved_solution, ls_sample_size)

        if improved_solution.cost < current.cost:
            current = improved_solution
            no_improve_count = 1
        else:
            no_improve_count += 1

    return current
