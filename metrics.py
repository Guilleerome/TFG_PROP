import instances_reader as ir
import constructor as construct
import solution as sol
import time
import local_Search as ls
import xlsxwriter
from itertools import permutations
from typing import Dict, Any, List
import numpy as np

class Metrics:
    def __init__(
        self,
        iteraciones: int = 60,
        excel_path: str = "experiments_results.xlsx",
        alphas=None
    ):
        if alphas is None:
            alphas = [0.25, 0.5, 0.75, 1.0]
        self.iteraciones = iteraciones
        self.excel_path = excel_path
        self.alphas = alphas

        self.plants = ir.read_instances()
        self.results: Dict[str, Any] = {}

        inicio = time.time()
        self.run_experiments()
        self.write_results_to_excel()
        fin = time.time()
        print(f"Tiempo total de métricas: {fin - inicio:.2f}s")


    def run_experiments(self):
        results = {}
        iteraciones = 60

        for plant in self.plants:
            print(f"Procesando planta: {plant.name}")

            # Registrar tiempos
            times = {"Guillermo": [], "random": [], "greedy_random_by_row": [], "random_greedy":[], "best_initial_solution" : 0, "local_search": {}, "iterative": 0}

            start_time = time.time()
            guillermo_solution = construct.construct_guillermo(plant)
            times["Guillermo"] = (time.time() - start_time)

            # Greedy random by row con diferentes valores de alfa y tiempos
            greedy_random_by_row_solutions, random_solutions, random_greedy_solutions = [], [], []
            greedy_random_by_row_times, random_times, random_greedy_times = [], [], []

            for alpha in self.alphas:
                greedy_random_by_row_time_acc = 0  # Acumulador de tiempo para Greedy random by row
                random_greedy_time_acc = 0  # Acumulador de tiempo para Random Greedy

                for _ in range(iteraciones):
                    # Greedy random by row
                    start_time = time.time()
                    greedy_random_by_row_solution = construct.constructor_greedy_random_by_row(plant, alpha)
                    greedy_random_by_row_time_acc += time.time() - start_time
                    greedy_random_by_row_solutions.append((alpha, greedy_random_by_row_solution))

                    # Random Greedy
                    start_time = time.time()
                    random_greedy_solution = construct.constructor_random_greedy(plant, alpha)
                    random_greedy_time_acc += time.time() - start_time
                    random_greedy_solutions.append((alpha, random_greedy_solution))

                # Promediar los tiempos para Greedy random by row y Random Greedy
                greedy_random_by_row_times.append(greedy_random_by_row_time_acc / iteraciones)
                random_greedy_times.append(random_greedy_time_acc / iteraciones)

            # Ejecución del Random
            random_time_acc = 0  # Acumulador de tiempo para Random

            for _ in range(self.iteraciones):
                start_time = time.time()
                random_solution = construct.construct_random(plant)
                random_time_acc += time.time() - start_time
                random_solutions.append(random_solution)

            # Promedio de tiempo para Random
            random_times = random_time_acc / iteraciones

            # Guardar los tiempos en la estructura `times`
            times["greedy_random_by_row"] = greedy_random_by_row_times
            times["random"] = random_times
            times["random_greedy"] = random_greedy_times

            # Calcular medias, desviaciones y conteos de best
            all_solution_costs = [guillermo_solution.cost] + \
                                 [solution.cost for solution in random_solutions] + \
                                 [solution.cost for _, solution in greedy_random_by_row_solutions] + \
                                 [solution.cost for _, solution in random_greedy_solutions]

            best_cost = min(all_solution_costs)  # Mejor costo global

            def calculate_std_dev(solutions):
                std_devs = [solution / best_cost for solution in solutions]
                mean_std_devs = np.mean(std_devs)
                return mean_std_devs

            # Guillermo estadísticas
            guillermo_avg_cost = guillermo_solution.cost
            guillermo_std_dev = calculate_std_dev([guillermo_solution.cost])
            guillermo_num_bests = 1 if guillermo_solution.cost == best_cost else 0

            greedy_random_by_row_avg_costs, greedy_random_by_row_std_devs, greedy_random_by_row_num_bests = {}, {}, {}
            #Greedy random by row estadísticas
            for alpha in self.alphas:
                solutions = [solution.cost for a, solution in greedy_random_by_row_solutions if a == alpha]
                greedy_random_by_row_avg_costs[alpha] = np.mean(solutions)
                greedy_random_by_row_std_devs[alpha] = calculate_std_dev(solutions)
                greedy_random_by_row_num_bests[alpha] = sum(1 for cost in solutions if cost == best_cost)

            random_greedy_avg_costs, random_greedy_std_devs, random_greedy_num_bests = {}, {}, {}
            # Random Greedy estadísticas
            for alpha in self.alphas:
                solutions = [solution.cost for a, solution in random_greedy_solutions if a == alpha]
                random_greedy_avg_costs[alpha] = np.mean(solutions)
                random_greedy_std_devs[alpha] = calculate_std_dev(solutions)
                random_greedy_num_bests[alpha] = sum(1 for cost in solutions if cost == best_cost)

            # Random estadísticas
            random_costs = [solution.cost for solution in random_solutions]
            random_avg_cost = np.mean(random_costs)
            random_std_dev = calculate_std_dev(random_costs)
            random_num_bests = sum(1 for cost in random_costs if cost == best_cost)

            # Seleccionar mejor solución inicial
            start_time = time.time()

            # Crear una lista extendida con información de origen para cada solución
            all_solutions = (
                    [(solution, f"Greedy Random by Row") for _, solution in greedy_random_by_row_solutions] +
                    [(solution, f"Random Greedy") for _, solution in random_greedy_solutions] +
                    [(solution, f"Random") for solution in random_solutions]
            )
            all_solutions.append((guillermo_solution, "Guillermo"))
            # Ordenar las soluciones por costo
            all_solutions_sorted = sorted(all_solutions, key=lambda s: s[0].cost)

            # Seleccionar las mejores soluciones, incluyendo empates
            best_solutions = []
            ordinal_map = ["1º", "2º", "3º"]  # Mapear posiciones ordinales
            current_rank = 0

            for solution, origin in all_solutions_sorted:
                if not best_solutions or solution.cost == best_solutions[-1][0].cost:
                    # Si la lista está vacía o el costo es igual al último agregado
                    if len(best_solutions) < len(ordinal_map):
                        best_solutions.append((solution, origin, ordinal_map[current_rank]))
                else:
                    # Si el costo es diferente, avanzar al siguiente ordinal
                    if current_rank < len(ordinal_map) - 1:
                        current_rank += 1
                        best_solutions.append((solution, origin, ordinal_map[current_rank]))
                # Si hemos llenado todas las posiciones posibles, detener
                if len(best_solutions) == len(ordinal_map):
                    break

            # Verificar si `guillermo_solution` está entre las mejores
            guillermo_in_best = any(sol == guillermo_solution for sol, _, _ in best_solutions)
            if not guillermo_in_best:
                # Identificar el ordinal de la última posición en ordinal_map
                last_position = ordinal_map[-1]

                # Remover cualquier solución que esté actualmente en esa última posición
                best_solutions = [
                    (sol, orig, rank)
                    for sol, orig, rank in best_solutions
                    if rank != last_position
                ]
                best_solutions.append((guillermo_solution, "Guillermo", ordinal_map[current_rank]))

            # Crear nombres con la posición correspondiente
            best_solutions_named = {
                f"{origin} ({position})": solution for solution, origin, position in best_solutions
            }

            end_time = time.time()
            times["best_initial_solution"] = end_time - start_time

            # Ejecutar búsquedas locales por separado sobre las mejores soluciones
            local_search_results = {}
            extended_local_search_results = {}

            local_search_methods = {
                "first_move_swap": ls.first_move_swap,
                "best_move_swap": ls.best_move_swap,
                "first_move": ls.first_move,
                "best_move": ls.best_move
            }

            # Ejecutar búsquedas locales individuales sobre cada solución en las mejores soluciones
            for name, solution in best_solutions_named.items():
                local_search_results[name] = {}
                for key, method in local_search_methods.items():
                    start_time = time.time()
                    result = method(solution)
                    local_search_results[name][key] = {
                        "cost": result.cost,
                        "time": time.time() - start_time
                    }

            # Ejecutar combinaciones de búsquedas locales
            valid_combinations = [
                ("first_move", "first_move_swap"),
                ("first_move", "best_move_swap"),
                ("best_move", "first_move_swap"),
                ("best_move", "best_move_swap"),
                ("first_move_swap", "first_move"),
                ("first_move_swap", "best_move"),
                ("best_move_swap", "first_move"),
                ("best_move_swap", "best_move")
            ]

            for name, solution in best_solutions_named.items():
                extended_local_search_results[name] = {}
                for combo in valid_combinations:
                    start_time = time.time()

                    intermediate_result = local_search_methods[combo[0]](solution)
                    final_result = local_search_methods[combo[1]](intermediate_result)

                    extended_local_search_results[name][" -> ".join(combo)] = {
                        "cost": final_result.cost,
                        "time": time.time() - start_time
                    }

            # # Iterative Local Search con tiempo
            # start_time = time.time()
            # final_best_solution = iterative_local_search(plant, best_initial_solution)
            # times["iterative"] = time.time() - start_time

            # Registrar resultados en un diccionario
            self.results[plant.name] = {
                "constructors": {
                    "guillermo": {
                        "best": guillermo_solution,
                        "average": guillermo_avg_cost,
                        "std_devs": guillermo_std_dev,
                        "num_bests": guillermo_num_bests,
                        "time": times["Guillermo"]
                    },
                    "greedy_random_by_row": {
                        "best": min([solution for _, solution in greedy_random_by_row_solutions]),
                        "averages": greedy_random_by_row_avg_costs,  # Coste promedio para cada alfa
                        "std_devs": greedy_random_by_row_std_devs,  # Desviación estándar para cada alfa
                        "num_bests": greedy_random_by_row_num_bests,  # Veces que greedy_random_by_row encuentra la mejor solución para cada alfa
                        "times": greedy_random_by_row_times  # Tiempos promedio por alfa
                    },
                    "random": {
                        "best": min([solution for solution in random_solutions]),
                        "average": random_avg_cost,  # Coste promedio
                        "std_devs": random_std_dev,  # Desviación estándar
                        "num_bests": random_num_bests,  # Veces que Random encuentra la mejor solución
                        "time": times["random"]  # Tiempo promedio
                    },
                    "random_greedy": {
                        "best": min([solution for _, solution in random_greedy_solutions]),
                        "averages": random_greedy_avg_costs,  # Coste promedio para cada alfa
                        "std_devs": random_greedy_std_devs,  # Desviación estándar para cada alfa
                        "num_bests": random_greedy_num_bests,
                        # Veces que Random Greedy encuentra la mejor solución para cada alfa
                        "times": random_greedy_times  # Tiempos promedio por alfa
                    }
                },
                "best_initial": {
                    "solutions": [sol.cost for sol, _, _ in best_solutions],  # Costes de las mejores soluciones iniciales
                    "time": times["best_initial_solution"]  # Tiempo para determinar las mejores soluciones iniciales
                },
                "local_search": {
                    "individual": local_search_results,  # Resultados de búsquedas locales individuales
                    "extended": extended_local_search_results  # Resultados de combinaciones de búsquedas locales
                },
                "times": times  # Todos los tiempos de ejecución registrados
            }

    def write_results_to_excel(self):
        # Crear el archivo Excel
        workbook = xlsxwriter.Workbook(self.excel_path, {"in_memory": True})

        # Formatos
        header_format = workbook.add_format({"bold": True, "bg_color": "#D7E4BC", "border": 1, "align": "center"})
        cell_format = workbook.add_format({"border": 1})
        cell_wrap_format = workbook.add_format({"border": 1})
        bold_format = workbook.add_format({"bold": True})

        # Ajustar ancho de columnas
        default_column_width = 25

        # Tabla 1: Resultados generales por planta
        worksheet_general = workbook.add_worksheet("Resultados Generales")
        headers_general = [
            "Planta", "Constructivo Guillermo (coste)",
            "Constructivo Random (Promedio)",
            "Constructivo Random Greedy (α=0.25)", "Constructivo Random Greedy (α=0.5)",
            "Constructivo Random Greedy (α=0.75)", "Constructivo Random Greedy (α=1.0)",
            "Constructivo Greedy Random by Row (α=0.25)", "Constructivo Greedy Random by Row (α=0.5)",
            "Constructivo Greedy Random by Row (α=0.75)", "Constructivo Greedy Random by Row (α=1.0)",
            "Mejores Soluciones Iniciales"
        ]
        worksheet_general.write_row(0, 0, headers_general, header_format)
        worksheet_general.set_column(0, len(headers_general) - 1, default_column_width)

        for row_idx, (plant_name, plant_results) in enumerate(self.results.items(), start=1):
            general_data = [
                plant_name,
                plant_results["constructors"]["guillermo"]["average"],
                plant_results["constructors"]["random"]["average"],
                plant_results["constructors"]["random_greedy"]["averages"].get(0.25, "N/A"),
                plant_results["constructors"]["random_greedy"]["averages"].get(0.5, "N/A"),
                plant_results["constructors"]["random_greedy"]["averages"].get(0.75, "N/A"),
                plant_results["constructors"]["random_greedy"]["averages"].get(1.0, "N/A"),
                plant_results["constructors"]["greedy_random_by_row"]["averages"].get(0.25, "N/A"),
                plant_results["constructors"]["greedy_random_by_row"]["averages"].get(0.5, "N/A"),
                plant_results["constructors"]["greedy_random_by_row"]["averages"].get(0.75, "N/A"),
                plant_results["constructors"]["greedy_random_by_row"]["averages"].get(1.0, "N/A"),
                " || ".join(map(str, plant_results["best_initial"]["solutions"])) if plant_results["best_initial"]["solutions"] else "N/A"
            ]
            worksheet_general.write_row(row_idx, 0, general_data, cell_format)

        # Tabla 2: Best constructors
        worksheet = workbook.add_worksheet("Best constructivo")
        constructores = ["guillermo", "random", "random_greedy", "greedy_random_by_row"]
        headers = ["Instancia"] + constructores + ["Best"]

        # Formatos
        cell_format = workbook.add_format({"border": 1, "align": "center"})
        highlight_format = workbook.add_format({"border": 1, "bold": True, "bg_color": "#FFD700", "align": "center"})  # Negrita y color de fondo dorado

        worksheet.write_row(0, 0, headers, header_format)
        worksheet.set_column(0, len(headers) - 1, 15)  # Ajusta ancho de columnas

        best_count = {constructor: 0 for constructor in constructores}

        row = 1
        for instancia, data in self.results.items():
            # Extraer los valores "best" por constructor
            best_solutions = {constructor: data["constructors"][constructor]["best"] for constructor in constructores}
            best_value = min(best_solutions.values())

            # Contabilizar el número de mejores valores
            for constructor, valor in best_solutions.items():
                if valor.cost == best_value.cost:
                    best_count[constructor] += 1

            # Preparar fila con valores
            fila = [instancia] + [best_solutions[constructor].cost for constructor in constructores] + [best_value.cost]

            # Escribir la fila y resaltar los valores mínimos
            for col, value in enumerate(fila):
                if col > 0 and col <= len(constructores):  # Solo para columnas de constructores
                    if value == best_value.cost:
                        worksheet.write(row, col, value, highlight_format)
                    else:
                        worksheet.write(row, col, value, cell_format)
                else:
                    worksheet.write(row, col, value, cell_format)

            row += 1

        # Escribir resumen
        worksheet.write(row, 0, "Resumen", bold_format)
        for col, constructor in enumerate(constructores, start=1):
            worksheet.write(row, col, best_count[constructor], cell_format)

        # Tabla 3: Resultados de búsquedas locales
        worksheet_local_search = workbook.add_worksheet("Búsquedas Locales")
        headers_local = ["Planta", "Soluciones", "Método", "coste", "Tiempo"]
        worksheet_local_search.write_row(0, 0, headers_local, header_format)
        worksheet_local_search.set_column(0, len(headers_local) - 1, default_column_width)

        row_idx = 1
        for plant_name, plant_results in self.results.items():
            for solution_id, search_results in plant_results["local_search"]["individual"].items():
                for method, metrics in search_results.items():
                    local_search_data = [
                        plant_name,
                        solution_id,
                        method,
                        metrics.get("cost", "N/A"),
                        metrics.get("time", "N/A")
                    ]
                    worksheet_local_search.write_row(row_idx, 0, local_search_data, cell_wrap_format)
                    row_idx += 1

        # Tabla 4: Mejores resultados de búsquedas locales
        worksheet_best_local = workbook.add_worksheet("Best Local Search")
        headers_best_local = ["Planta", "Mejores Soluciones", "Métodos Locales", "coste"]
        worksheet_best_local.write_row(0, 0, headers_best_local, header_format)
        worksheet_best_local.set_column(0, len(headers_best_local) - 1, default_column_width)

        local_methods = ["first_move_swap", "best_move_swap", "first_move", "best_move"]
        best_count_local = {method: 0 for method in local_methods}
        best_solution_count = {}

        row = 1
        for plant_name, plant_results in self.results.items():
            initial_cost = plant_results["best_initial"]["solutions"][0] # Costo de la solución inicial

            best_cost = float('inf')
            best_solutions = []
            best_methods = []

            for solution_id, search_results in plant_results["local_search"]["individual"].items():
                for method, metrics in search_results.items():
                    current_cost = metrics.get("cost", float('inf'))
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_solutions = [solution_id]
                        best_methods = [method]
                    elif current_cost == best_cost:
                        best_solutions.append(solution_id)
                        best_methods.append(method)

            # Si ninguna búsqueda local mejora la solución inicial
            if best_cost >= initial_cost:
                best_constructor , _ = next(iter(plant_results["local_search"]["individual"].items()))
                worksheet_best_local.write_row(
                    row,
                    0,
                    [plant_name, best_constructor, "Ninguna búsqueda mejora la solución", initial_cost],
                    cell_format,
                )
            else:
                worksheet_best_local.write_row(
                    row,
                    0,
                    [plant_name, ", ".join(best_solutions), ", ".join(best_methods), best_cost],
                    cell_format,
                )
                for method in best_methods:
                    best_count_local[method] += 1
                for solution in best_solutions:
                    best_solution_count[solution] = best_solution_count.get(solution, 0) + 1

            row += 1

        row += 2
        worksheet_best_local.write(row, 0, "Resumen Métodos", bold_format)
        row += 1
        worksheet_best_local.write_row(row, 0, ["Método", "Veces mejor", "Tiempos Medios"], header_format)
        row += 1

        # Diccionario para acumular tiempos
        method_times = {method: 0 for method in local_methods}

        for plant_name, plant_results in self.results.items():
            for solution_id, search_results in plant_results["local_search"]["individual"].items():
                for method, metrics in search_results.items():
                    if metrics.get("cost", float('inf')) == best_cost:
                        method_times[method] += metrics.get("time", 0)

        for method, count in best_count_local.items():
            avg_time = method_times[method] / count if count > 0 else 0
            worksheet_best_local.write_row(row, 0, [method, count, avg_time], cell_format)
            row += 1

        # Resumen de mejores soluciones
        row += 2
        worksheet_best_local.write(row, 0, "Resumen Soluciones", bold_format)
        row += 1
        worksheet_best_local.write_row(row, 0, ["Solución", "Veces mejor"], header_format)
        row += 1
        for solution, count in best_solution_count.items():
            worksheet_best_local.write_row(row, 0, [solution, count], cell_format)
            row += 1

        # Tabla 5: Resultados de búsquedas locales extendidas
        worksheet_extended_search = workbook.add_worksheet("Búsquedas Extendidas")
        headers_extended = ["Planta", "Solución", "Combinación", "coste", "Tiempo"]
        worksheet_extended_search.write_row(0, 0, headers_extended, header_format)
        worksheet_extended_search.set_column(0, len(headers_extended) - 1, default_column_width)

        row_idx = 1
        for plant_name, plant_results in self.results.items():
            for solution_id, extended_search_results in plant_results["local_search"]["extended"].items():
                for combination, metrics in extended_search_results.items():
                    extended_search_data = [
                        plant_name,
                        solution_id,
                        combination,
                        metrics.get("cost", "N/A"),
                        metrics.get("time", "N/A")
                    ]
                    worksheet_extended_search.write_row(row_idx, 0, extended_search_data, cell_wrap_format)
                    row_idx += 1

        # Tabla 6: Mejores resultados de búsquedas extendidas
        worksheet_best_extended = workbook.add_worksheet("Best Extended Search")
        headers_best_extended = ["Planta", "Mejores Soluciones", "Combinaciones", "coste"]
        worksheet_best_extended.write_row(0, 0, headers_best_extended, header_format)
        worksheet_best_extended.set_column(0, len(headers_best_extended) - 1, default_column_width)

        best_count_extended = {}
        best_solution_usage_count = {}
        extended_method_times = {combination: 0 for combination in best_count_extended}  # Diccionario para acumular tiempos
        row = 1

        for plant_name, plant_results in self.results.items():
            initial_cost = plant_results["best_initial"]["solutions"][0]  # Costo de la solución inicial
            best_cost = float('inf')
            best_solutions = []
            best_combinations = []

            for solution_id, extended_search_results in plant_results["local_search"]["extended"].items():
                for combination, metrics in extended_search_results.items():
                    current_cost = metrics.get("cost", float('inf'))
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_solutions = [solution_id]
                        best_combinations = [combination]
                    elif current_cost == best_cost:
                        best_solutions.append(solution_id)
                        best_combinations.append(combination)

            # Acumular tiempos solo para las combinaciones que resultaron mejores
            for solution_id, extended_search_results in plant_results["local_search"]["extended"].items():
                for combination, metrics in extended_search_results.items():
                    if metrics.get("cost", float('inf')) == best_cost:  # Combinación que coincide con el mejor costo
                        if combination not in extended_method_times:
                            extended_method_times[combination] = metrics.get("time", 0)
                        else:
                            extended_method_times[combination] += metrics.get("time", 0)

            # Si ninguna búsqueda extendida mejora la solución inicial
            if best_cost >= initial_cost:
                best_constructor, _ = next(iter(plant_results["local_search"]["individual"].items()))
                worksheet_best_extended.write_row(
                    row,
                    0,
                    [plant_name, best_constructor, "Ninguna búsqueda mejora la solución", initial_cost],
                    cell_format,
                )
            else:
                worksheet_best_extended.write_row(
                    row,
                    0,
                    [plant_name, ", ".join(best_solutions), ", ".join(best_combinations), best_cost],
                    cell_format,
                )
                for combination in best_combinations:
                    best_count_extended[combination] = best_count_extended.get(combination, 0) + 1
                for solution in best_solutions:
                    best_solution_usage_count[solution] = best_solution_usage_count.get(solution, 0) + 1

            row += 1

        # Resumen de combinaciones con tiempos medios
        row += 2
        worksheet_best_extended.write(row, 0, "Resumen Combinaciones", bold_format)
        row += 1
        worksheet_best_extended.write_row(row, 0, ["Combinación", "Veces mejor", "Tiempos Medios"], header_format)
        row += 1

        for combination, count in best_count_extended.items():
            avg_time = extended_method_times[combination] / count if count > 0 else 0
            worksheet_best_extended.write_row(row, 0, [combination, count, avg_time], cell_format)
            row += 1

        # Resumen de mejores soluciones extendidas
        row += 2
        worksheet_best_extended.write(row, 0, "Resumen Soluciones", bold_format)
        row += 1
        worksheet_best_extended.write_row(row, 0, ["Solución", "Veces mejor"], header_format)
        row += 1

        for solution, count in best_solution_usage_count.items():
            worksheet_best_extended.write_row(row, 0, [solution, count], cell_format)
            row += 1

        # Hoja de resultados generales
        worksheet_summary = workbook.add_worksheet("Estadísticas Generales")
        headers_summary = ["Algoritmo", "Media (Coste)", "Media (Tiempo)", "Desviación Típica", "Cantidad de Best"]
        worksheet_summary.write_row(0, 0, headers_summary, header_format)
        worksheet_summary.set_column(0, len(headers_summary) - 1, default_column_width)

        row_idx = 1

        # Agregar estadísticas de constructores
        aggregated_constructors = self.aggregate_statistics_across_instances(self.results, "constructors")
        for method_name, method_data in aggregated_constructors.items():
            avg_cost, avg_time, std_dev, total_bests = self.calculate_overall_statistics([method_data])
            worksheet_summary.write_row(
                row_idx, 0,
                [f"Constructivo - {method_name}", avg_cost, avg_time, std_dev, total_bests],
                cell_format
            )
            row_idx += 1

        # # Agregar estadísticas de búsquedas locales individuales
        # aggregated_individual_search = aggregate_statistics_across_instances(results, "local_search")
        # for method_name, method_data in aggregated_individual_search.get("individual", {}).items():
        #     avg_cost, avg_time, std_dev, total_bests = calculate_overall_statistics([method_data])
        #     worksheet_summary.write_row(
        #         row_idx, 0,
        #         [f"Búsqueda Local - {method_name}", avg_cost, avg_time, std_dev, total_bests],
        #         cell_format
        #     )
        #     row_idx += 1
        #
        # # Agregar estadísticas de búsquedas locales extendidas
        # aggregated_extended_search = aggregate_statistics_across_instances(results, "local_search")
        # for method_name, method_data in aggregated_extended_search.get("extended", {}).items():
        #     avg_cost, avg_time, std_dev, total_bests = calculate_overall_statistics([method_data])
        #     worksheet_summary.write_row(
        #         row_idx, 0,
        #         [f"Búsqueda Local Extendida - {method_name}", avg_cost, avg_time, std_dev, total_bests],
        #         cell_format
        #     )
        #     row_idx += 1

        workbook.close()

    def calculate_overall_statistics(self, data):
        """
        Calcula las estadísticas globales (media, tiempo promedio, desviación típica y número de best)
        a partir de los datos de todas las instancias.
        """
        all_costs = []
        all_times = []
        all_std_dev = []
        total_bests = 0

        for instance_data in data:
            all_costs.extend(instance_data.get("costs", []))
            all_times.extend(instance_data.get("times", []))
            all_std_dev.extend(instance_data.get("std_devs", []))
            total_bests += instance_data.get("num_bests", 0)

        avg_cost = np.mean(all_costs) if all_costs else 0
        avg_time = np.mean(all_times) if all_times else 0
        std_dev = np.mean(all_std_dev) if all_std_dev else 0

        return avg_cost, avg_time, std_dev, total_bests


    def aggregate_statistics_across_instances(self, results, key):
        """
        Agrega estadísticas para todas las instancias agrupadas por metodo (por ejemplo, constructor o búsqueda local).
        """
        aggregated = {}

        for plant_results in results.values():
            methods = plant_results.get(key, {})

            if key == "constructors":
                # Procesar constructores como greedy_random_by_row y random_greedy
                for method_name, method_data in methods.items():
                    if method_name in ["greedy_random_by_row", "random_greedy"]:
                        # Procesar métodos con alfas
                        averages = method_data.get("averages", {})
                        if not averages:  # Saltar si no hay datos en "averages"
                            continue

                        std_devs = method_data.get("std_devs", {})
                        num_bests = method_data.get("num_bests", {})
                        times = method_data.get("times", [])

                        # Ordenar los alfas
                        ordered_alphas = sorted(averages.keys())

                        for alpha, avg_cost in averages.items():
                            if alpha not in ordered_alphas:
                                continue  # Saltar si el alfa no es válido
                            alpha_index = ordered_alphas.index(alpha)

                            # Obtener valores correspondientes
                            avg_time = times[alpha_index] if alpha_index < len(times) else 0
                            std_dev = std_devs.get(alpha, 0)
                            num_best = num_bests.get(alpha, 0)

                            alpha_name = f"{method_name} (alpha {alpha})"
                            if alpha_name not in aggregated:
                                aggregated[alpha_name] = {
                                    "costs": [],
                                    "times": [],
                                    "num_bests": 0,
                                    "std_devs": []
                                }

                            aggregated[alpha_name]["costs"].append(avg_cost)
                            aggregated[alpha_name]["times"].append(avg_time)
                            aggregated[alpha_name]["num_bests"] += num_best
                            aggregated[alpha_name]["std_devs"].append(std_dev)
                    else:
                        # Procesar otros métodos
                        if method_name not in aggregated:
                            aggregated[method_name] = {
                                "costs": [],
                                "times": [],
                                "num_bests": 0,
                                "std_devs": []
                            }

                        aggregated[method_name]["costs"].append(method_data.get("average", 0))
                        aggregated[method_name]["times"].append(method_data.get("time", 0))
                        aggregated[method_name]["num_bests"] += method_data.get("num_bests", 0)
                        aggregated[method_name]["std_devs"].append(method_data.get("std_devs", 0))

            elif key == "local_search":
                # Procesar búsquedas locales (individual y extended)
                for search_type, search_methods in methods.items():
                    for solution_name, methods in search_methods.items():
                        search_key = f"{search_type} - {solution_name}"

                        if search_key not in aggregated:
                            aggregated[search_key] = {
                                "costs": [],
                                "times": [],
                                "num_bests": 0
                            }

                        for method_name, method_results in methods.items():
                            costs = method_results.get("costs", [])
                            times = method_results.get("times", [])
                            num_bests = method_results.get("num_bests", 0)

                            aggregated[search_key]["costs"].extend(costs)
                            aggregated[search_key]["times"].extend(times)
                            aggregated[search_key]["num_bests"] += num_bests

        # Limpiar entradas vacías (opcional)
        aggregated = {k: v for k, v in aggregated.items() if v["costs"]}

        return aggregated
