import zipfile
from pathlib import Path
import re

def restructure_instances_parallel_row_benchmarks_zip(zip_path: Path):
    zip_path = Path(zip_path)
    out_dir = zip_path.with_suffix('')
    out_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        for member in z.namelist():
            if not member.lower().endswith('.txt'):
                continue

            raw = z.read(member).decode('utf-8')
            # Extraer todos los enteros
            nums = list(map(int, re.findall(r'-?\d+', raw)))

            idx = 0
            N = nums[idx];
            idx += 1  # total de facilities
            F = nums[idx];
            idx += 1  # número de filas (se asume 2)
            if F != 2:
                raise ValueError(f"{member}: se esperaban 2 filas, pero F={F}")

            # Capacidades de cada fila
            row_caps = nums[idx: idx + F]
            idx += F
            if len(row_caps) != 2:
                raise ValueError(f"{member}: no hay dos capacidades de fila")

            # Tamaños de cada facility
            sizes = nums[idx: idx + N]
            idx += N
            if len(sizes) != N:
                raise ValueError(f"{member}: faltan tamaños (esperados {N}, hallados {len(sizes)})")

            # Matriz de costes plana
            total_costs = N * N
            flat_costs = nums[idx: idx + total_costs]
            idx += total_costs
            if len(flat_costs) != total_costs:
                raise ValueError(f"{member}: faltan costes (esperados {total_costs}, hallados {len(flat_costs)})")

            # Reconstruir la matriz N×N
            matrix = [flat_costs[i * N:(i + 1) * N] for i in range(N)]

            # Guardar en el formato estándar
            dest = out_dir / Path(member).name
            with open(dest, 'w', encoding='utf-8') as fout:
                fout.write(f"{N}\n")
                fout.write(" ".join(map(str, sizes)) + "\n")
                fout.write(f"{row_caps[0]} {row_caps[1]}\n")
                for row in matrix:
                    fout.write(" ".join(map(str, row)) + "\n")

    print(f"Instancias re-estructuradas en: {out_dir}")

def restructure_instances_prop_instances(zip_path: Path):
    zip_path = Path(zip_path)
    out_dir = zip_path.with_suffix('')
    out_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        for member in z.namelist():
            if not member.lower().endswith('.txt'):
                continue

            raw = z.read(member).decode('utf-8')
            tokens = raw.split()

            # 1) filas y N total
            n = int(tokens[0])
            fila1 = int(tokens[1])
            fila2 = n - fila1
            if fila2 < 0:
                raise ValueError(f"{member}: fila1 ({fila1}) > N ({n})")

            # 2) Los siguientes N tokens son los tamaños de cada facility
            if len(tokens) < 2 + n:
                raise ValueError(f"{member}: no hay suficientes tokens para los tamaños (necesito {n})")
            sizes = list(map(int, tokens[2:2 + n]))

            # 3) Tras los tamaños vienen N*N tokens de la matriz de costes
            start = 2 + n
            end = start + n * n
            if len(tokens) < end:
                raise ValueError(f"{member}: no hay suficientes tokens para la matriz de costes ({n * n} valores)")
            flat_costs = list(map(int, tokens[start:end]))

            # Reconstruir la matriz N×N
            matrix = [flat_costs[i * n:(i + 1) * n] for i in range(n)]

            # 4) Escribir el nuevo .txt
            dest = out_dir / Path(member).name
            with open(dest, 'w', encoding='utf-8') as fout:
                # Línea 1: N total
                fout.write(f"{n}\n")
                # Línea 2: tamaños en una única línea
                fout.write(" ".join(map(str, sizes)) + "\n")
                # Línea 3: fila1 fila2
                fout.write(f"{fila1} {fila2}\n")
                # Líneas 4–(4+N-1): matriz de costes
                for row in matrix:
                    fout.write(" ".join(map(str, row)) + "\n")

            print(f"Instancias re-estructuradas en: {out_dir}")