import zipfile
from pathlib import Path

import plant as plant_module

def read_all_instances():
    base = Path("instances")
    all_plants = []
    for sub in sorted(base.iterdir()):
        if sub.is_dir():
            all_plants.extend(read_instances(sub.name))
    return all_plants

def read_instances(folder_name: str):
    base_dir = Path("instances") / folder_name
    plants = []
    for file_path in sorted(base_dir.glob("*.txt")):
        with file_path.open("r", encoding="utf-8") as f:
            n = int(f.readline().strip())
            #capacity = n / rows
            facilities = list(map(int, f.readline().strip().split()))
            capacities = list(map(int, f.readline().strip().split()))
            # Crear un diccionario con n entradas
            # facilities = {i: valores[i] for i in range(n)}

            # Inicializar una lista para la matriz
            distances = []

            # Leer el resto del archivo para llenar la matriz
            for linea in f:
                linea_matriz = linea.strip()
                if linea_matriz:
                    fila = list(map(int, linea_matriz.strip().split()))
                    distances.append(fila)
            instance_name = file_path.stem
            # distances = make_matrix_symmetric(distances)
            plants.append(plant_module.Plant(instance_name, n, len(capacities), capacities, facilities, distances))
    return plants

def make_matrix_symmetric(m):
    n = len(m)
    for i in range(n):
        for j in range(i + 1, n):  # Solo completar la parte inferior
            if j < len(m[i]):
                m[j][i] = m[i][j]  # Hacer que sea simétrica
            else:
                # Si faltan columnas en las filas inferiores, las creamos
                m[j].append(m[i][j])
    return m



