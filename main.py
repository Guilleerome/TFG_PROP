import instances_reader as ir
import constructor as construct
import time
import local_search as ls
from metrics import Metrics
from concurrent.futures import ProcessPoolExecutor

NUM_CONSTRUCTIONS = 10  # veces que se aplica el constructor
ALFA = 0.3
SAMPLE_SIZE = 40

def generate_and_improve_solution_for_plant(args):
    plant, alfa, sample_size = args
    sol_inicial = construct.constructor_greedy_random_by_row(plant, alfa=alfa, sample_size=sample_size)
    sol_mejorada = ls.swap_then_first_one_by_one(sol_inicial)
    return sol_mejorada

def process_plant(plant):
    args = [(plant, ALFA, SAMPLE_SIZE)] * NUM_CONSTRUCTIONS
    with ProcessPoolExecutor() as executor:
        soluciones = list(executor.map(generate_and_improve_solution_for_plant, args))

    mejor_sol = min(soluciones, key=lambda s: s.cost)

    print(f"Plant: {plant.name}, Cost: {mejor_sol.cost}")
    return mejor_sol

def run_parallel(plants):
    with ProcessPoolExecutor() as executor:
        soluciones_finales = list(executor.map(process_plant, plants))

def run_serial(plants):
    for plant in plants:
        soluciones_finales = []
        for _ in range(NUM_CONSTRUCTIONS):
            sol_inicial = construct.constructor_greedy_random_global(plant, alfa=ALFA, sample_size=SAMPLE_SIZE)
            sol_mejorada = ls.swap_then_first_one_by_one(sol_inicial)
            soluciones_finales.append(sol_mejorada)

        mejor_sol = min(soluciones_finales, key=lambda s: s.cost)
        print(f"Plant: {plant.name}, Cost: {mejor_sol.cost}")

def main():
    plants = ir.read_instances("Small")

    start_time = time.time()
    run_serial(plants)
    end_time = time.time()
    print(f"Tiempo total: {end_time - start_time:.2f} segundos")

if __name__ == "__main__":
    main()