import instances_reader as ir
import constructor as construct
import Solution as sol
import time
import Local_Search as ls
import xlsxwriter


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
            current_solution = best_local_solution
            improved = True

    return current_solution


def run_experiments():
    plants = ir.read_instances()
    results = {}

    for plant in plants:
        print(f"Procesando planta: {plant.name}")

        # Registrar tiempos
        times = {"deterministic": [], "grasp": [], "best_initial_solution" : 0, "local_search": {}, "iterative": 0}

        # Constructores deterministas con tiempos
        deterministic_solutions = []
        for constructor in [construct.construct_random, construct.construct_greedy, construct.construct_greedy_2]:
            start_time = time.time()
            solution = constructor(plant)
            times["deterministic"].append(time.time() - start_time)
            deterministic_solutions.append(solution)

        # GRASP con diferentes valores de alfa y tiempos
        grasp_alphas = [0, 0.25, 0.5, 0.75, 1]
        grasp_solutions = []
        grasp_times = []

        for alpha in grasp_alphas:
            start_time = time.time()
            for _ in range(50):
                grasp_solution = construct.constructor_grasp(plant, alpha)
                grasp_solutions.append((alpha, grasp_solution))
            grasp_times.append((time.time() - start_time) / 50)  # Promedio por iteración

        times["grasp"] = grasp_times

        # Calcular medias de GRASP
        grasp_avg_costs = {}
        for alpha in grasp_alphas:
            solutions = [solution.cost for a, solution in grasp_solutions if a == alpha]
            grasp_avg_costs[alpha] = sum(solutions) / len(solutions)

        # Seleccionar mejor solución inicial
        start_time = time.time()
        all_solutions = deterministic_solutions + [solution for _, solution in grasp_solutions]
        best_initial_solution = min(all_solutions, key=lambda s: s.cost)
        end_time = time.time()
        times["best_initial_solution"] = start_time - end_time

        # Ejecutar búsquedas locales por separado con tiempos
        local_search_results = {}
        local_search_methods = {
            "first_move_swap": ls.first_move_swap,
            "best_move_swap": ls.best_move_swap,
            "first_move": ls.first_move,
            "best_move": ls.best_move
        }

        for key, method in local_search_methods.items():
            start_time = time.time()
            result = method(best_initial_solution)
            local_search_results[key] = result.cost
            times["local_search"][key] = time.time() - start_time

        # Iterative Local Search con tiempo
        start_time = time.time()
        final_best_solution = iterative_local_search(plant, best_initial_solution)
        times["iterative"] = time.time() - start_time

        # Registrar resultados en un diccionario
        results[plant.name] = {
            "constructors": {
                "deterministic": [sol.cost for sol in deterministic_solutions],
                "grasp": grasp_avg_costs
            },
            "best_initial": best_initial_solution.cost,
            "local_search": local_search_results,
            "final_best": final_best_solution.cost,
            "times": times
        }

    return plants, results


def write_results_to_excel(plants, results):
    # Crear el archivo Excel
    workbook = xlsxwriter.Workbook("experiment_results.xlsx", {"in_memory": True})
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

    # Fila adicional para tiempos medios
    times_row = len(results) + 2
    worksheet.write(times_row, 0, "Tiempos Medios", header_format)

    avg_times = []

    # Promedios de tiempos para constructores deterministas
    for i in range(3):  # Tres constructores deterministas
        avg_times.append(
            sum(result["times"]["deterministic"][i] for _, result in results.items()) / len(results)
        )

    # Promedios de tiempos para GRASP (5 configuraciones de alfa)
    for i in range(5):
        avg_times.append(
            sum(result["times"]["grasp"][i] for _, result in results.items()) / len(results)
        )

    avg_times.append(
        sum(result["times"]["best_initial_solution"] for _, result in results.items()) / len(results)
    )

    # Tiempo promedio de búsqueda local
    for key in ["first_move_swap", "best_move_swap", "first_move", "best_move"]:
        avg_times.append(
            sum(result["times"]["local_search"][key] for _, result in results.items()) / len(results)
        )

    # Tiempo promedio para Iterative Local Search
    avg_times.append(
        sum(result["times"]["iterative"] for _, result in results.items()) / len(results)
    )

    # Escribir tiempos en las columnas correspondientes
    worksheet.write_row(times_row, 1, avg_times, cell_format)

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