import instances_reader as ir
import constructor as construct
import Solution as sol
import time
import Local_Search as ls
import xlsxwriter


def run_experiments():
    plants = ir.read_instances()
    results = {}

    for plant in plants:
        print(f"Procesando planta: {plant.name}")

        # Constructores deterministas
        deterministic_solutions = [
            construct.construct_random(plant),
            construct.construct_greedy(plant),
            construct.construct_greedy_2(plant)
        ]

        # GRASP con diferentes valores de alfa
        grasp_alphas = [0, 0.25, 0.5, 0.75, 1]
        grasp_solutions = []

        for alpha in grasp_alphas:
            for _ in range(50):
                grasp_solution = construct.constructor_grasp(plant, alpha)
                grasp_solutions.append((alpha, grasp_solution))  # Guardar alfa junto con la solución

        # Calcular medias de GRASP
        grasp_avg_costs = {}
        for alpha in grasp_alphas:
            # Filtrar soluciones por alfa y calcular promedio
            solutions = [solution.cost for a, solution in grasp_solutions if a == alpha]
            grasp_avg_costs[alpha] = sum(solutions) / len(solutions)

        # Seleccionar mejor solución inicial
        all_solutions = deterministic_solutions + [solution for _, solution in grasp_solutions]
        best_initial_solution = min(all_solutions, key=lambda s: s.cost)
        print(f"Mejor solución inicial: {best_initial_solution.cost}")

        # Ejecutar búsquedas locales por separado
        local_search_results = {
            "first_move_swap": ls.first_move_swap(best_initial_solution),
            "best_move_swap": ls.best_move_swap(best_initial_solution),
            "first_move": ls.first_move(best_initial_solution),
            "best_move": ls.best_move(best_initial_solution)
        }

        # Iterative Local Search
        final_best_solution = iterative_local_search(plant, best_initial_solution)

        # Registrar resultados en un diccionario
        results[plant.name] = {
            "constructors": {
                "deterministic": [sol.cost for sol in deterministic_solutions],
                "grasp": grasp_avg_costs
            },
            "best_initial": best_initial_solution.cost,
            "local_search": {key: sol.cost for key, sol in local_search_results.items()},
            "final_best": final_best_solution.cost
        }

    return plants, results

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

def write_results_to_excel(plants, results):
    # Crear el archivo Excel
    workbook = xlsxwriter.Workbook("experiment_results.xlsx", {'in_memory':True})
    worksheet = workbook.add_worksheet("Resultados")

    # Formatos
    header_format = workbook.add_format({"bold": True, "bg_color": "#D7E4BC", "border": 1, "align": "center"})
    cell_format = workbook.add_format({"border": 1})

    # Cabeceras
    headers = [
        "Planta",
        "Constructor Random",
        "Constructor Greedy",
        "Constructor Greedy 2",
        "GRASP (α=0)",
        "GRASP (α=0.25)",
        "GRASP (α=0.5)",
        "GRASP (α=0.75)",
        "GRASP (α=1)",
        "Mejor Solución Inicial",
        "Búsqueda Local (FM Swap)",
        "Búsqueda Local (BM Swap)",
        "Búsqueda Local (FM)",
        "Búsqueda Local (BM)",
        "Mejor Solución Final"
    ]
    worksheet.write_row(0, 0, headers, header_format)

    # Ajustar anchos de las columnas
    column_width = 15
    for col_idx in range(len(headers)):
        worksheet.set_column(col_idx, col_idx, column_width)

    # Escribir resultados
    for row_idx, (plant_name, plant_results) in enumerate(results.items(), start=1):
        processed_results = [
            plant_results["constructors"]["deterministic"][0],  # Random
            plant_results["constructors"]["deterministic"][1],  # Greedy
            plant_results["constructors"]["deterministic"][2],  # Greedy 2
            plant_results["constructors"]["grasp"][0],         # GRASP (α=0)
            plant_results["constructors"]["grasp"][0.25],      # GRASP (α=0.25)
            plant_results["constructors"]["grasp"][0.5],       # GRASP (α=0.5)
            plant_results["constructors"]["grasp"][0.75],      # GRASP (α=0.75)
            plant_results["constructors"]["grasp"][1],         # GRASP (α=1)
            plant_results["best_initial"],                    # Mejor Solución Inicial
            plant_results["local_search"]["first_move_swap"],  # FM Swap
            plant_results["local_search"]["best_move_swap"],   # BM Swap
            plant_results["local_search"]["first_move"],       # FM
            plant_results["local_search"]["best_move"],        # BM
            plant_results["final_best"]                        # Mejor Solución Final
        ]

        worksheet.write(row_idx, 0, plant_name, cell_format)  # Nombre de la planta
        worksheet.write_row(row_idx, 1, processed_results, cell_format)  # Resultados

    workbook.close()

def main():
    inicio = time.time()

    plants, results = run_experiments()

    # Generar archivo Excel
    write_results_to_excel(plants, results)
    fin = time.time()
    print(f"Tiempo total de ejecución: {fin - inicio} segundos")

if __name__ == "__main__":
    main()
