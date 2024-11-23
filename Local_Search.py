import random
import Solution as sol
from copy import deepcopy

import random
import copy

def first_move(solution):
    best_solution = solution
    improved = True

    while improved:
        improved = False
        new_solution = sol.Solution(plant=solution.plant, cost=float('inf'))

        # Generar el orden aleatorio para filas y movimientos
        order_moves = []
        order_rows = []

        for i in range(len(solution.disposition)):
            aux = solution.disposition[i][::]
            random.shuffle(aux)
            order_moves.append(aux)
            order_rows.append(i)
        random.shuffle(order_rows)

        # Probar todos los movimientos posibles
        for i in order_rows:
            for j in range(len(order_moves[i])):
                for k in range(j + 1, len(order_moves[i])):
                    # Crear una copia de la disposición actual
                    disposition_aux = copy.deepcopy(solution.disposition)

                    # Intercambiar las posiciones de las facilities
                    disposition_aux[i][j], disposition_aux[i][k] = disposition_aux[i][k], disposition_aux[i][j]

                    # Evaluar la nueva solución
                    new_solution.changeDisposition(disposition_aux)
                    if new_solution < best_solution:
                        print(f"Nueva mejor solución encontrada. Costo: {new_solution.cost}")
                        # Actualizar la mejor solución y reiniciar búsqueda
                        best_solution = new_solution
                        improved = True
                        break  # Salir de los bucles internos para reiniciar desde el principio
                if improved:
                    break
            if improved:
                break

    return best_solution

    
        
