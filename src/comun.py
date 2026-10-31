"""Utilidades compartidas por los pasos del pipeline."""
from pathlib import Path
import re
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MANUAL = ROOT / "data" / "manual"
FIGURAS = ROOT / "outputs" / "figuras"


def leer_csv(path, **kw):
    """Lee un CSV en UTF-8 (con BOM) y, si falla, en latin-1."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", **kw)


def normalizar_nombre(s):
    """Normaliza un nombre de municipio: sin acentos, minúsculas y solo alfanuméricos."""
    s = str(s).replace("*", "").strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def a_numero(s):
    return pd.to_numeric(s, errors="coerce")


class Bitacora:
    """Imprime mensajes y los acumula para guardarlos en un archivo de texto."""

    def __init__(self):
        self.lineas = []

    def __call__(self, *partes):
        texto = " ".join(str(p) for p in partes)
        print(texto)
        self.lineas.append(texto)

    def guardar(self, path):
        Path(path).write_text("\n".join(self.lineas), encoding="utf-8")
