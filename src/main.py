from io_instances import instances_reader as ir
from constructors import constructor as construct
import time
from improvers import local_search as ls
from models.solution import Solution
from models.plant import Plant
from concurrent.futures import ProcessPoolExecutor
from config import ExperimentConfig as cfg


def generate_and_improve_solution_for_plant(args: tuple[Plant, float, int]) -> Solution:
    plant, alfa, sample_size = args
    sol_initial = construct.constructor_greedy_random_by_row(plant, alfa=alfa, sample_size=sample_size)
    sol_improved = ls.swap_then_first_one_by_one(sol_initial)
    return sol_improved

def process_plant(plant: Plant) -> Solution:
    args = [(plant, cfg.ALPHA, cfg.SAMPLE_SIZE)] * cfg.NUM_CONSTRUCTIONS
    with ProcessPoolExecutor() as executor:
        soluciones = list(executor.map(generate_and_improve_solution_for_plant, args))

    mejor_sol = min(soluciones, key=lambda s: s.cost)

    print(f"Plant: {plant.name}, Cost: {mejor_sol.cost}")
    return mejor_sol

def run_parallel(plants: list[Plant]) -> None:
    with ProcessPoolExecutor() as executor:
        soluciones_finales = list(executor.map(process_plant, plants))

def run_serial(plants: list[Plant]) -> None:
    for plant in plants:
        final_solutions = []
        for _ in range(cfg.NUM_CONSTRUCTIONS):
            sol_initial = construct.constructor_greedy_random_global(plant, alfa=cfg.ALPHA, sample_size=cfg.SAMPLE_SIZE)
            sol_improved = ls.swap_then_first_one_by_one(sol_initial)
            final_solutions.append(sol_improved)

        best_sol = min(final_solutions, key=lambda s: s.cost)
        print(f"Plant: {plant.name}, Cost: {best_sol.cost}")

def main() -> None:
    plants = ir.read_instances(cfg.INSTANCE_FOLDER)

    start_time = time.time()
    run_serial(plants)
    end_time = time.time()
    print(f"Tiempo total: {end_time - start_time:.2f} segundos")

if __name__ == "__main__":
    main()