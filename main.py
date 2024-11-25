import instances_reader as ir
import constructor as construct
import Solution as sol
import time
import improver as imp
import Local_Search as ls

def iterative_local_search(plant, initial_solution):

    current_solution = initial_solution
    improved = True

    while improved:
        improved = False

        fm_swap_solution = ls.first_move_swap(current_solution)
        bm_swap_solution = ls.best_move_swap(current_solution)
        fm_solution = ls.first_move(current_solution)
        bm_solution = ls.best_move(current_solution)

        best_local_solution = min([fm_swap_solution, bm_swap_solution, fm_solution, bm_solution], key=lambda s: s.cost)

        if best_local_solution.cost < current_solution.cost:
            print(f"Mejora encontrada. Costo: {best_local_solution.cost}")
            current_solution = best_local_solution
            improved = True

    return current_solution

def main():
    inicio = time.time()

    plants = ir.read_instances()

    for plant in plants:
        print(f"Planta: {plant.name}")

        solutions = [
            construct.construct_random(plant),
            construct.construct_greedy(plant),
            construct.construct_greedy_2(plant),
            construct.constructor_grasp(plant, 0.5)
        ]

        solutions = [imp.improve_greedy(sol) for sol in solutions]

        best_initial_solution = min(solutions, key=lambda s: s.cost)
        print(f"Mejor solución inicial de constructores - Costo: {best_initial_solution.cost}")
        print(best_initial_solution.disposition)

        final_solution = iterative_local_search(plant, best_initial_solution)

        print("Solución final tras búsqueda local:")
        print(final_solution.disposition)
        print(f"Coste final: {final_solution.cost}")
        print("")

    fin = time.time()
    print(f"Tiempo de ejecución: {fin - inicio} segundos")

if __name__ == "__main__":
    main()
