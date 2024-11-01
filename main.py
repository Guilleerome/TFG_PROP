import instances_reader as ir
import constructor as construct
import Solution as sol
import time

inicio = time.time()

plants = ir.read_instances()

for plant in plants:
    print(f"Planta: {plant.name}")

    solution = construct.construct_random(plant)
    print(f"construct_random - Costo: {solution.cost}")

    solution = construct.construct_greedy(plant)
    print(f"construct_greedy - Costo: {solution.cost}")

    solution = construct.construct_greedy_2(plant)
    print(f"construct_greedy_2 - Costo: {solution.cost}")
    print(solution.disposition)
    print("")

fin = time.time()
print(f"Tiempo de ejecución: {fin - inicio}")