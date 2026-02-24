from io_instances import instances_reader as ir
import time
from models.solution import Solution
from models.plant import Plant
from concurrent.futures import ProcessPoolExecutor
from config import ExperimentConfig as cfg
from src.utils import Metrics, MetricsTop3ConstructorsLocalSearch, MetricsPaperComparisonTop3
from algorithms import run_algorithm, run_grasp


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
    for alpha in [0.8, 1.0]:
        for plant in plants[0:15]:
            final_solutions = []
            for _ in range(cfg.NUM_CONSTRUCTIONS):
                run = run_algorithm('grasp', plant, constructor_name='greedy_random_by_row',
                                    ls_sequence=['best_move_swap', 'first_move'], alpha=alpha, sample_size=cfg.SAMPLE_SIZE)
                final_solutions.append(run)
            best_sol = min(final_solutions, key=lambda s: s.cost)
            print(f"Plant: {plant.name}, Cost: {best_sol.cost}")


def main() -> None:
    plants = ir.read_instances(cfg.INSTANCE_FOLDER)
    start_time = time.time()
    # Metrics("results_for_Nico3", plants=plants2, iterations=30)
    # MetricsTop3ConstructorsLocalSearch(plants=plants2, iterations=30)
    #MetricsPaperComparisonTop3(plants=plants, iterations=30, sota_csv_path="../Papers/sota_ma.csv")
    run_serial(plants)
    end_time = time.time()
    print(f"Tiempo total: {end_time - start_time:.2f} segundos")

if __name__ == "__main__":
    main()