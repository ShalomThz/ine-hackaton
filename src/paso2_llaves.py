"""
Paso 2 · Consolidación de la CCPC y equivalencia de claves municipales INE ↔ INEGI.

Entradas (data/raw/)
    DatosAbiertos_DECEyEC_ConteosCensales<AÑO>/datosabiertos_*.csv   CCPC 2009-2024
    conjunto_de_datos_eic2025*/conjunto_de_datos/*.csv               Intercensal 2025

Salidas (data/processed/)
    ccpc_mun_sexo_edad.csv.gz   LN, SV, NV y NS por año × municipio INE × sexo × edad
    ccpc_mun_tiposec.csv        LN por año × municipio INE × tipo de sección (U/M/R)
    eic2025_mun.csv             Intercensal 2025, totales municipales (estimador "Valor")
    equivalencias_mun.csv       clave INE → clave INEGI (CVEGEO de 5 dígitos)
    equivalencias_revisar.csv   empates no exactos o ausentes, para revisión manual
    catalogo_ine_cambios.csv    claves INE cuyo nombre cambia entre elecciones
    inegi_sin_ine.csv           municipios INEGI sin clave INE asignada
    excel/*.csv                 hojas de los Excel de apoyo (ECEG, CONAPO, CONEVAL)

Uso (desde la raíz del proyecto)
    python src/paso2_llaves.py
"""
import difflib
import re
import time

import pandas as pd

from comun import PROC, RAW, leer_csv, normalizar_nombre

COLUMNAS_CCPC = ["AELEC", "EDOCVE", "MPIOCVE", "MPIONOM", "TIPOSEC",
                 "SEXO", "EDAD", "LN", "SV", "NV", "NS"]
COLUMNAS_NUMERICAS = ["AELEC", "EDOCVE", "MPIOCVE", "SEXO", "EDAD", "LN", "SV", "NV", "NS"]
LLAVE_EDAD = ["AELEC", "EDOCVE", "MPIOCVE", "SEXO", "EDAD"]
LLAVE_TIPO = ["AELEC", "EDOCVE", "MPIOCVE", "TIPOSEC"]
CONTEOS = ["LN", "SV", "NV", "NS"]

LONGITUD_MIN_CONTENIDO = 4
UMBRAL_APROXIMADO = 0.80


def consolidar_ccpc():
    archivos = sorted(RAW.glob("DatosAbiertos_DECEyEC_ConteosCensales*/datosabiertos_*.csv"))
    print(f"Archivos CCPC encontrados: {len(archivos)}")

    por_edad, por_tipo, nombres, llaves_nulas = [], [], [], []
    inicio = time.time()
    for i, archivo in enumerate(archivos, 1):
        t = time.time()
        df = leer_csv(archivo, usecols=COLUMNAS_CCPC,
                      dtype={"MPIONOM": "string", "TIPOSEC": "string"},
                      low_memory=False)
        for c in COLUMNAS_NUMERICAS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["TIPOSEC"] = df["TIPOSEC"].str.strip().str[0].str.upper()

        nulos = df[LLAVE_EDAD].isna().any(axis=1).sum()
        if nulos:
            llaves_nulas.append((archivo.name, int(nulos)))

        por_edad.append(df.groupby(LLAVE_EDAD, as_index=False)[CONTEOS].sum())
        por_tipo.append(df.groupby(LLAVE_TIPO, as_index=False)["LN"].sum())
        nombres.append(df[["AELEC", "EDOCVE", "MPIOCVE", "MPIONOM"]].drop_duplicates())
        print(f"[{i:3d}/{len(archivos)}] {archivo.name}: {len(df):,} filas ({time.time() - t:.1f}s)")

    ccpc = pd.concat(por_edad, ignore_index=True).groupby(LLAVE_EDAD, as_index=False).sum()
    ccpc.to_csv(PROC / "ccpc_mun_sexo_edad.csv.gz", index=False, compression="gzip")

    tipo = pd.concat(por_tipo, ignore_index=True).groupby(LLAVE_TIPO, as_index=False).sum()
    tipo.to_csv(PROC / "ccpc_mun_tiposec.csv", index=False)

    print(f"\nCCPC consolidada en {time.time() - inicio:.0f}s -> {len(ccpc):,} filas")
    if llaves_nulas:
        print("Filas con llaves nulas (descartadas):", llaves_nulas)

    return ccpc, pd.concat(nombres, ignore_index=True).drop_duplicates()


def resumir_ccpc(ccpc):
    res = ccpc.groupby("AELEC")[CONTEOS].sum()
    res["part_SV_LN"] = (res.SV / res.LN * 100).round(1)
    res["part_sin_NS"] = (res.SV / (res.SV + res.NV) * 100).round(1)
    res["municipios"] = (ccpc[["AELEC", "EDOCVE", "MPIOCVE"]]
                         .drop_duplicates().groupby("AELEC").size())
    print("\nResumen por elección (contrastar con el tablero del INE):\n", res.to_string())
    print("\nLista nominal por sexo y elección:\n",
          ccpc.pivot_table(index="AELEC", columns="SEXO", values="LN", aggfunc="sum").to_string())


def construir_catalogo_ine(nombres):
    cat = nombres.copy()
    cat["norm"] = cat["MPIONOM"].map(normalizar_nombre)

    n_nombres = cat.groupby(["EDOCVE", "MPIOCVE"])["norm"].nunique()
    claves_con_cambio = n_nombres[n_nombres > 1].index
    if len(claves_con_cambio):
        cambios = cat.set_index(["EDOCVE", "MPIOCVE"]).loc[claves_con_cambio].reset_index()
    else:
        cambios = cat.iloc[0:0]
    cambios.sort_values(["EDOCVE", "MPIOCVE", "AELEC"]).to_csv(
        PROC / "catalogo_ine_cambios.csv", index=False)

    cat_ine = (cat.sort_values("AELEC")
                  .groupby(["EDOCVE", "MPIOCVE"])
                  .agg(nom_ine=("MPIONOM", "last"),
                       anios=("AELEC", lambda s: ",".join(map(str, sorted(set(s))))))
                  .reset_index())
    cat_ine["norm"] = cat_ine["nom_ine"].map(normalizar_nombre)
    print(f"\nMunicipios INE distintos: {len(cat_ine):,} "
          f"(claves con cambio de nombre: {len(claves_con_cambio)})")
    return cat_ine


def cargar_intercensal():
    ruta = next(RAW.glob("conjunto_de_datos_eic2025*/conjunto_de_datos/*.csv"))
    eic = leer_csv(ruta, dtype=str)
    eic = eic[(eic["ESTIMADOR"].str.strip() == "Valor")
              & (eic["CVE_LOC"] == "0000")
              & (~eic["CVE_MUN"].isin(["000", "997"]))].copy()
    eic["NOM_MUN"] = eic["NOM_MUN"].str.replace("*", "", regex=False).str.strip()
    eic["CVEGEO_MUN"] = eic["CVE_ENT"] + eic["CVE_MUN"]
    eic.to_csv(PROC / "eic2025_mun.csv", index=False)

    inegi = eic[["CVE_ENT", "CVE_MUN", "NOM_MUN", "CVEGEO_MUN"]].copy()
    inegi["CVE_ENT"] = inegi["CVE_ENT"].astype(int)
    inegi["norm"] = inegi["NOM_MUN"].map(normalizar_nombre)
    print(f"Municipios INEGI (EIC 2025): {len(inegi):,}")
    return inegi


def empatar_nombre(nombre, por_nombre):
    if nombre in por_nombre:
        return "exacto", por_nombre[nombre], 1.0

    contenidos = [n for n in por_nombre
                  if (nombre in n or n in nombre)
                  and min(len(n), len(nombre)) >= LONGITUD_MIN_CONTENIDO]
    if len(contenidos) == 1:
        score = difflib.SequenceMatcher(None, nombre, contenidos[0]).ratio()
        return "contenido", por_nombre[contenidos[0]], score

    parecidos = difflib.get_close_matches(nombre, list(por_nombre), n=1, cutoff=UMBRAL_APROXIMADO)
    if parecidos:
        score = difflib.SequenceMatcher(None, nombre, parecidos[0]).ratio()
        return "aproximado", por_nombre[parecidos[0]], score

    return "sin_empate", None, 0.0


def empatar_ine_inegi(cat_ine, inegi):
    filas = []
    for edo, grupo in cat_ine.groupby("EDOCVE"):
        candidatos = inegi[inegi["CVE_ENT"] == edo]
        por_nombre = dict(zip(candidatos["norm"], candidatos["CVEGEO_MUN"]))
        nom_inegi = dict(zip(candidatos["CVEGEO_MUN"], candidatos["NOM_MUN"]))
        for _, r in grupo.iterrows():
            metodo, cve, score = empatar_nombre(r["norm"], por_nombre)
            filas.append({"EDOCVE": edo, "MPIOCVE": r["MPIOCVE"], "nom_ine": r["nom_ine"],
                          "anios_ine": r["anios"], "CVEGEO_MUN": cve,
                          "nom_inegi": nom_inegi.get(cve), "metodo": metodo,
                          "score": round(score, 3)})

    eq = pd.DataFrame(filas)
    duplicados = eq["CVEGEO_MUN"].notna() & eq.duplicated("CVEGEO_MUN", keep=False)
    eq["revisar"] = (eq["metodo"] != "exacto") | duplicados
    eq.to_csv(PROC / "equivalencias_mun.csv", index=False)
    eq[eq["revisar"]].to_csv(PROC / "equivalencias_revisar.csv", index=False)

    sin_ine = inegi[~inegi["CVEGEO_MUN"].isin(eq["CVEGEO_MUN"])]
    sin_ine.to_csv(PROC / "inegi_sin_ine.csv", index=False)

    print("\nResultado del empate:\n", eq["metodo"].value_counts().to_string())
    print(f"INEGI asignados a más de una clave INE: {duplicados.sum()}")
    print(f"Municipios INEGI sin clave INE: {len(sin_ine)}")
    print(f"Por revisar a mano: {eq['revisar'].sum()} -> data/processed/equivalencias_revisar.csv")


def exportar_excels_a_csv():
    destino_dir = PROC / "excel"
    destino_dir.mkdir(exist_ok=True)
    excels = (list(RAW.glob("eceg_2020*/catalogos/cat_municipios_*.xlsx"))
              + list(RAW.glob("IMM_2020*.xlsx"))
              + list(RAW.glob("Concentrado_indicadores_de_pobreza*.xlsx")))
    for libro in excels:
        try:
            hojas = pd.read_excel(libro, sheet_name=None, header=None, dtype=str)
        except ImportError:
            print("\nPara leer los Excel instala openpyxl: pip install openpyxl")
            return
        for nombre, hoja in hojas.items():
            destino = destino_dir / f"{libro.stem}__{re.sub(r'[^A-Za-z0-9]+', '_', str(nombre))}.csv"
            hoja.to_csv(destino, index=False, header=False)
            print(f"\n{libro.name} | hoja '{nombre}': {hoja.shape[0]:,} filas x {hoja.shape[1]} cols"
                  f" -> excel/{destino.name}")
            print(hoja.head(8).to_string(max_colwidth=30))


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    ccpc, nombres = consolidar_ccpc()
    resumir_ccpc(ccpc)
    cat_ine = construir_catalogo_ine(nombres)
    inegi = cargar_intercensal()
    empatar_ine_inegi(cat_ine, inegi)
    exportar_excels_a_csv()
    print("\nListo.")


if __name__ == "__main__":
    main()
