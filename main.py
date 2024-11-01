import instances_reader as ir
import constructor as construct
import Solution as sol
import time

plants = ir.read_instances()

for plant in plants:
    print(f"Planta: {plant.name}")

    inicio = time.time()
    solution = construct.construct_random(plant)
    fin = time.time()
    print(f"construct_random - Costo: {solution.cost}, Tiempo: {fin - inicio:.4f} segundos")

    inicio = time.time()
    solution = construct.construct_greedy(plant)
    fin = time.time()
    print(f"construct_greedy - Costo: {solution.cost}, Tiempo: {fin - inicio:.4f} segundos")

    inicio = time.time()
    solution = construct.construct_greedy_2(plant)
    fin = time.time()
    print(f"construct_greedy_2 (False) - Costo: {solution.cost}, Tiempo: {fin - inicio:.4f} segundos")
    print(solution.disposition)
    print("")

