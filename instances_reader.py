import zipfile
import Plant as plant_module

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

zip_file = zipfile.ZipFile('instances/small.zip', 'r')
rows = 2
plants = []

def read_instances():
    for file in zip_file.namelist():
        if file.endswith('.txt'):
            with zip_file.open(file) as f:
                n = int(f.readline().strip())
                #capacity = n / rows
                valores = list(map(int, f.readline().strip().split()))
                # Crear un diccionario con n entradas
                facilities = {}
                #facilities_distribution = []
                facilities = {i: valores[i] for i in range(n)}
                '''for j in range(rows - 1):
                    facilities_distribution.append(
                        {(i + j * rows): valores[(i + j * rows)] for i in range(capacity.__floor__())})
                facilities_distribution.append({i: valores[i] for i in range(n - capacity.__ceil__(), n)})'''

                # Inicializar una lista para la matriz
                distances = []

                # Leer el resto del archivo para llenar la matriz
                for linea in f:
                    linea_matriz = linea.strip()
                    if linea_matriz:
                        fila = list(map(int, linea_matriz.strip().split()))
                        distances.append(fila)
                instance_name = file.split('/')[1].split('.')[0]
                # distances = make_matrix_symmetric(distances)
                plants.append(plant_module.Plant(instance_name, n, rows, facilities, distances))
    zip_file.close()
    return plants





