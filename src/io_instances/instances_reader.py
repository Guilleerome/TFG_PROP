import os
from pathlib import Path
from typing import List
from src.models.plant import Plant


def _resolve_instances_root(explicit_root: str | Path | None = None) -> Path:
    """Devuelve la ruta absoluta a 'instances' buscando de forma robusta."""
    if explicit_root:
        p = Path(explicit_root).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"No existe la carpeta de instancias: {p}")
        return p

    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "instances",  # W:\...\TFG_PROP\instances
        here.parents[1] / "instances",  # W:\...\TFG_PROP\src\instances
        here.parent / "instances",      # W:\...\TFG_PROP\src\io_instances\instances
        Path.cwd() / "instances",       # por si el CWD ya es la raíz
        Path(os.getenv("TFG_INSTANCES")) if os.getenv("TFG_INSTANCES") else None,
    ]
    tried = []
    for c in candidates:
        if c is None:
            continue
        c = c.resolve()
        tried.append(str(c))
        if c.exists():
            return c
    raise FileNotFoundError(
        "No se encontró carpeta 'instances'. Probadas:\n- " + "\n- ".join(tried)
    )
def read_all_instances(root: str | Path | None = None) -> List[Plant]:
    base = _resolve_instances_root(root)
    all_plants = []
    for sub in sorted(base.iterdir()):
        if sub.is_dir():
            all_plants.extend(read_instances(sub.name))
    return all_plants

def read_instances(folder_name: str, root: str | Path | None = None) -> List[Plant]:
    base_dir = _resolve_instances_root(root) / folder_name
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
            plants.append(Plant(instance_name, n, len(capacities), capacities, facilities, distances))
    return plants

def make_matrix_symmetric(m: List[List[int]]) -> List[List[int]]:
    n = len(m)
    for i in range(n):
        for j in range(i + 1, n):  # Solo completar la parte inferior
            if j < len(m[i]):
                m[j][i] = m[i][j]  # Hacer que sea simétrica
            else:
                # Si faltan columnas en las filas inferiores, las creamos
                m[j].append(m[i][j])
    return m



