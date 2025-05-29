import instances_reader as ir
import constructor as construct
import solution as sol
import time
import local_Search as ls
import xlsxwriter
from itertools import permutations
import numpy as np
import metrics


def main():

    plants = ir.read_instances()
    results = []

    for plant in plants:
        start_time = time.time()
        best_solution = None
        for alfa in [0.25, 0.5, 0.75, 1]:
            i = 0
            # Construct initial solution
            initial_solution = construct.constructor_random_greedy(plant, alfa)

            improved_solution = ls.best_move_swap(initial_solution)
            improved_solution = ls.first_move(improved_solution)

            if i == 0:
                best_solution = improved_solution
            else:
                if improved_solution < best_solution:
                    best_solution = improved_solution
            i += 1
        elapsed_time = time.time() - start_time
        results.append((plant.name, best_solution.cost, elapsed_time))

        print(f"Plant: {plant.name}, Cost: {best_solution.cost}, Time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()