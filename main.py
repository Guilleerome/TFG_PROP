import instances_reader as ir
import constructor as construct
import Solution as sol
import time
import improver as imp
import Local_Search as ls

inicio = time.time()

plants = ir.read_instances()

for plant in plants:
    print(f"Planta: {plant.name}")

    solution = construct.construct_random(plant)
    print(f"construct_random - Costo: {solution.cost}")
    solution = imp.improve_greedy(solution)
    print(f"improve_greedy - Costo: {solution.cost}")
    print(solution.disposition)
    print("")

    solution = construct.construct_greedy(plant)
    print(f"construct_greedy - Costo: {solution.cost}")
    solution = imp.improve_greedy(solution)
    print(f"improve_greedy - Costo: {solution.cost}")
    print(solution.disposition)
    print("")

    solution = construct.construct_greedy_2(plant)
    print(f"construct_greedy_2 - Costo: {solution.cost}")
    print(solution.disposition)

    solution = imp.improve_greedy(solution)
    print(f"improve_greedy - Costo: {solution.cost}")
    print(solution.disposition)
    print("")

    solution = ls.first_move(solution)
    print(solution.disposition)
    print(f"local_serach - Cost: {solution.cost}")

fin = time.time()
print(f"Tiempo de ejecución: {fin - inicio} segundos")