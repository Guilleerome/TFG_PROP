#!/usr/bin/env python3
"""
run_irace.py - Ejecutar irace desde Python

Script para ejecutar irace con calibración completa de GRASP:
- Constructores
- Parámetros de constructores (alpha, sample_size)
- Búsquedas locales (secuencia)
- Parámetros de búsquedas locales (ls_sample_size)

Uso desde PyCharm:
    - Click derecho → Run 'run_irace'

Uso desde terminal:
    python run_irace.py
"""

import subprocess
import sys
import tempfile
import os
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN - Modifica estos valores según tus necesidades
# ============================================================================

# Configuración de irace
IRACE_CONFIG = {
    'maxExperiments': 10000,
    'firstTest': 5,
    'eachTest': 1,
    'confidence': 0.95,
    'testType': 'friedman',
    'parallel': 3,
    'seed': 123456,
}


# ============================================================================


def find_r_executable():
    """Encuentra Rscript.exe en ubicaciones comunes de Windows."""
    common_paths = [
        Path(r"C:\Program Files\R"),
        Path(r"C:\Program Files (x86)\R"),
    ]

    for base_path in common_paths:
        if not base_path.exists():
            continue

        for r_version_dir in sorted(base_path.glob("R-*"), reverse=True):
            rscript_path = r_version_dir / "bin" / "Rscript.exe"
            if rscript_path.exists():
                return rscript_path

    return None


def main():
    print("=" * 70)
    print("irace - Calibración Completa de GRASP")
    print("=" * 70)

    # Directorios
    irace_dir = Path(__file__).parent.resolve()
    project_root = irace_dir.parent.parent
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"

    print(f"\n📁 Directorio irace: {irace_dir}")
    print(f"🐍 Python: {venv_python}")

    # Verificar archivos necesarios
    print("\n✓ Verificando archivos...")
    files_to_check = {
        "target_runner.py": irace_dir / "target_runner.py",
        "parameters.txt": irace_dir / "parameters.txt",
        "instances/": irace_dir / "instances",
        "Python venv": venv_python,
    }

    all_exist = True
    for name, path in files_to_check.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {name}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\n❌ Error: Faltan archivos necesarios")
        return 1

    # Buscar Rscript
    print("\n🔍 Buscando R...")
    rscript_path = find_r_executable()

    if not rscript_path:
        print("❌ No se encontró Rscript.exe")
        print("\nPor favor, instala R desde: https://cran.r-project.org/")
        return 1

    print(f"✓ R encontrado: {rscript_path}")

    # Contar instancias
    instances = list((irace_dir / "instances").glob("*.txt"))
    print(f"\n📊 Instancias encontradas: {len(instances)}")

    # Generar código R
    print("\n⚙️  Generando configuración de irace...")

    # Convertir rutas a formato R (forward slashes)
    irace_dir_r = str(irace_dir).replace('\\', '/')
    venv_python_r = str(venv_python).replace('\\', '/')

    # Script R que lee parameters.txt
    r_script = f"""
library(irace)

cat("\\n")
cat("======================================================================\\n")
cat("Iniciando calibración completa de GRASP con irace\\n")
cat("======================================================================\\n\\n")

# Cambiar al directorio de trabajo
setwd("{irace_dir_r}")

# Leer parámetros desde parameters.txt
parameters <- readParameters(file = "parameters.txt")

cat("Parámetros a calibrar:\\n")
print(parameters$names)
cat("\\n")
cat("Número de parámetros:", length(parameters$names), "\\n\\n")

# Listar instancias
instance_files <- list.files("instances", pattern = "[.]txt$", full.names = TRUE)
instance_names <- basename(tools::file_path_sans_ext(instance_files))

instances <- instance_files
names(instances) <- instance_names

cat("Número de instancias:", length(instances), "\\n")
cat("Primeras instancias:", paste(head(instance_names, 5), collapse=", "), "...\\n\\n")

# Configurar scenario
scenario <- list(
  instances = instances,
  parameters = parameters,
  targetRunner = "target_runner.py",
  targetRunnerLauncher = "{venv_python_r}",

  maxExperiments = {IRACE_CONFIG['maxExperiments']},
  firstTest = {IRACE_CONFIG['firstTest']},
  eachTest = {IRACE_CONFIG['eachTest']},
  confidence = {IRACE_CONFIG['confidence']},
  testType = "{IRACE_CONFIG['testType']}",

  parallel = {IRACE_CONFIG['parallel']},
  loadBalancing = TRUE,
  mpi = FALSE,

  seed = {IRACE_CONFIG['seed']},
  deterministic = FALSE,

  execDir = getwd(),
  logFile = "irace.Rdata",

  elitist = TRUE,
  elitistNewInstances = 1,
  postselection = 1,

  capping = FALSE
)

# Ejecutar irace
cat("Ejecutando irace...\\n")
cat("Presupuesto: {IRACE_CONFIG['maxExperiments']} evaluaciones\\n")
cat("Paralelización: {IRACE_CONFIG['parallel']} cores\\n")
cat("\\n")

result <- irace(scenario = scenario)

cat("\\n")
cat("======================================================================\\n")
cat("Calibración completada\\n")
cat("======================================================================\\n")
cat("\\nMejores configuraciones encontradas:\\n")
print(result)
cat("\\nResultados guardados en: irace.Rdata\\n")
"""

    # Escribir el script R a un archivo temporal
    r_script_path = irace_dir / "_irace_run.R"
    r_script_path.write_text(r_script, encoding='utf-8')
    print(f"📝 Script R generado: {r_script_path}")

    # Ejecutar R con --file en lugar de -e (evita límite de longitud en Windows)
    print("\n🚀 Ejecutando irace...\n")
    print("=" * 70)

    try:
        result = subprocess.run(
            [str(rscript_path), "--file", str(r_script_path)],
            cwd=str(irace_dir),
            capture_output=False,
            text=True
        )

        print("=" * 70)

        if result.returncode == 0:
            print("\n✅ irace completado exitosamente!")
            print(f"\n📁 Resultados guardados en: {irace_dir / 'irace.Rdata'}")
            print("\nPara ver los resultados en R:")
            print("  load('irace.Rdata')")
            print("  print(iraceResults)")
        else:
            print(f"\n❌ irace terminó con errores (código {result.returncode})")
            print(f"\n💡 Puedes inspeccionar el script R generado en: {r_script_path}")

        return result.returncode

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Limpiar el archivo temporal si fue bien
        if result.returncode == 0 and r_script_path.exists():
            r_script_path.unlink()


if __name__ == "__main__":
    exit(main())