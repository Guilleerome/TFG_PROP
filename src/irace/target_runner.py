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

from io_instances import instances_reader as ir
from algorithms import run_algorithm


def parse_irace_arguments(args):
    """
    Parsea los argumentos de irace en un formato más manejable

    irace pasa argumentos en el formato:
    target_runner.py <config_id> <instance_id> <seed> <instance_path> [--param1 value1] [--param2 value2] ...

    Returns:
        dict con: config_id, instance_id, seed, instance_path, params
    """
    if len(args) < 5:
        raise ValueError(f"Argumentos insuficientes. Recibidos: {args}")

    result = {
        'config_id': args[1],
        'instance_id': args[2],
        'seed': int(args[3]),
        'instance_path': args[4],
        'params': {}
    }

    # Parsear parámetros adicionales (formato: --param value)
    i = 5
    while i < len(args):
        if args[i].startswith('--'):
            param_name = args[i][2:]  # Quitar '--'
            if i + 1 < len(args):
                param_value = args[i + 1]
                result['params'][param_name] = param_value
                i += 2
            else:
                i += 1
        else:
            i += 1

    return result


def convert_param_types(params):
    """
    Convierte los parámetros de string a sus tipos correctos

    Args:
        params: dict con parámetros como strings

    Returns:
        dict con parámetros en sus tipos correctos
    """
    converted = {}

    for key, value in params.items():
        # Parámetros numéricos
        if key in ['alpha', 'weight_flows']:
            converted[key] = float(value)
        elif key in ['sample_size', 'ls_sample_size']:
            converted[key] = int(value)
        # Parámetros categóricos (strings)
        elif key in ['constructor', 'ls1', 'ls2', 'algorithm']:
            converted[key] = value
        else:
            # Por defecto, dejar como string
            converted[key] = value

    return converted


def build_algorithm_params(params):
    """
    Construye los parámetros específicos para run_algorithm

    Args:
        params: dict con parámetros parseados

    Returns:
        dict con parámetros listos para run_algorithm
    """
    algo_params = {}

    # Tipo de algoritmo (por defecto GRASP)
    algorithm = params.get('algorithm', 'grasp')

    # Parámetros comunes de GRASP
    if 'constructor' in params:
        algo_params['constructor_name'] = params['constructor']

    if 'alpha' in params:
        algo_params['alpha'] = params['alpha']

    if 'sample_size' in params:
        algo_params['sample_size'] = params['sample_size']

    if 'ls_sample_size' in params:
        algo_params['ls_sample_size'] = params['ls_sample_size']

    # Búsquedas locales (construir secuencia)
    ls_sequence = []
    if 'ls1' in params and params['ls1'] != 'none':
        ls_sequence.append(params['ls1'])
    if 'ls2' in params and params['ls2'] != 'none':
        ls_sequence.append(params['ls2'])

    if ls_sequence:
        algo_params['ls_sequence'] = ls_sequence

    return algorithm, algo_params


def main():
    try:
        # Parsear argumentos
        parsed = parse_irace_arguments(sys.argv)

        config_id = parsed['config_id']
        instance_id = parsed['instance_id']
        seed = parsed['seed']
        instance_path = parsed['instance_path']
        params_raw = parsed['params']

        # Convertir tipos de parámetros
        params = convert_param_types(params_raw)

        # Establecer semilla
        random.seed(seed)
        np.random.seed(seed)

        # Leer instancia
        instance_path_obj = Path(instance_path)
        instance_folder = instance_path_obj.parent.name
        instance_name = instance_path_obj.stem
        instances_root = instance_path_obj.parent.parent

        plants = ir.read_instances(instance_folder, root=instances_root)

        plant = None
        for p in plants:
            if p.name == instance_name:
                plant = p
                break

        if plant is None:
            print(f"Error: Instancia {instance_name} no encontrada", file=sys.stderr)
            sys.exit(1)

        # Construir parámetros del algoritmo
        algorithm, algo_params = build_algorithm_params(params)

        # Ejecutar algoritmo
        start_time = time.time()

        solution = run_algorithm(
            algorithm_name=algorithm,
            plant=plant,
            **algo_params
        )

        total_time = time.time() - start_time

        # IMPORTANTE: irace espera UN SOLO NÚMERO en stdout
        print(solution.cost)

        # Debug comentado (descomentar solo para debugging, no para irace)
        # print(f"[DEBUG] Config={config_id} Instance={instance_id} "
        #       f"Params={params} Cost={solution.cost} Time={total_time:.3f}s",
        #       file=sys.stderr)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()