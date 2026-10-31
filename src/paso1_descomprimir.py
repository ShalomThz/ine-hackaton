"""
Paso 1 · Descomprime los datos fuente de data/zip/ en data/raw/.

Los datos crudos (~3.7 GB) se versionan comprimidos para respetar el límite de
tamaño de GitHub. Este paso los restaura antes de correr el resto del pipeline.

Uso (desde la raíz del proyecto)
    python src/paso1_descomprimir.py
"""
import zipfile

from comun import RAW, ROOT

ZIPS = ROOT / "data" / "zip"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    archivos = sorted(ZIPS.glob("*.zip"))
    if not archivos:
        raise SystemExit(f"No se encontraron archivos .zip en {ZIPS}")
    for i, archivo in enumerate(archivos, 1):
        print(f"[{i:2d}/{len(archivos)}] {archivo.name}")
        with zipfile.ZipFile(archivo) as z:
            z.extractall(RAW)
    print(f"\nListo. Datos en {RAW.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
