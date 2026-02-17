#!/usr/bin/env python3
"""
target_runner.py - Wrapper para irace

Ejecuta el algoritmo con los parámetros dados por irace e imprime el coste.
"""

import sys
import time
import random
import numpy as np
from pathlib import Path

# Agregar directorios necesarios al path
src_dir = Path(__file__).resolve().parent.parent  # src/
project_root = src_dir.parent  # TFG_PROP/
sys.path.insert(0, str(project_root))  # Para imports "from src.xxx"
sys.path.insert(0, str(src_dir))  # Para imports "from io_instances"

# Imports locales (ahora funcionan porque agregamos src/ al path)
from io_instances import instances_reader as ir
from constructors import constructor as construct
from improvers import local_search as ls


def main():
    # Verificar argumentos
    if len(sys.argv) < 7:
        print("Error: Argumentos insuficientes", file=sys.stderr)
        sys.exit(1)

    # Parsear argumentos de irace
    config_id = sys.argv[1]
    instance_id = sys.argv[2]
    seed = int(sys.argv[3])
    instance_path = sys.argv[4]
    alpha = float(sys.argv[5])
    sample_size = int(sys.argv[6])

    # Establecer semilla
    random.seed(seed)
    np.random.seed(seed)

    try:
        # Leer la instancia
        instance_path_obj = Path(instance_path)
        instance_folder = instance_path_obj.parent.name
        instance_name = instance_path_obj.stem
        instances_root = instance_path_obj.parent.parent

        plants = ir.read_instances(instance_folder, root=instances_root)

        # Buscar la planta correcta
        plant = None
        for p in plants:
            if p.name == instance_name:
                plant = p
                break

        if plant is None:
            print(f"Error: Instancia {instance_name} no encontrada", file=sys.stderr)
            sys.exit(1)

        # Ejecutar algoritmo
        start_time = time.time()

        sol_initial = construct.constructor_greedy_random_by_row(
            plant,
            alfa=alpha,
            sample_size=sample_size
        )

        #sol_improved = ls.best_move_swap(sol_initial)
        #sol_improved = ls.first_move(sol_improved)

        total_time = time.time() - start_time
        final_cost = sol_initial.cost

        # IMPORTANTE: irace lee SOLO esto de stdout
        print(final_cost)

        # Debug info comentado porque irace solo acepta un número en stdout
        # print(f"[DEBUG] Config={config_id} Instance={instance_id} "
        #       f"Alpha={alpha} Sample={sample_size} Cost={final_cost} Time={total_time:.3f}s",
        #       file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()