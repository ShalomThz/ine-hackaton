"""
Paso 3 · Base de análisis: participación electoral (CCPC) + contexto municipal.

Unidad de análisis: municipio × sexo × grupo de edad (Hombres/Mujeres; 18-24 … 75+).

Entradas
    data/processed/ccpc_mun_sexo_edad.csv.gz, ccpc_mun_tiposec.csv,
    data/processed/equivalencias_mun.csv, catalogo_ine_cambios.csv, eic2025_mun.csv
    data/processed/excel/IMM_2020__IMM_2020.csv                                   CONAPO
    data/processed/excel/Concentrado_indicadores_de_pobreza_2020__Concentrado_municipal.csv   CONEVAL
    data/raw/iter_00_cpv2020*/.../conjunto_de_datos_iter_00CSV20.csv              ITER 2020
    data/manual/correcciones_equivalencias.csv

Salidas (data/processed/)
    participacion_largo.csv   participación por celda y elección
    contexto_mun.csv          contexto socioeconómico por municipio (clave INEGI)
    base_analisis.csv         una fila por celda con participación y contexto
    validacion_paso3.txt      bitácora de decisiones y validaciones

Uso (desde la raíz del proyecto)
    python src/paso3_base_analisis.py
"""
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from comun import MANUAL, PROC, RAW, Bitacora, a_numero, leer_csv, normalizar_nombre

PRESIDENCIALES = [2012, 2018, 2024]
INTERMEDIAS = [2009, 2015, 2021]
EDAD_BINS = [17, 24, 34, 44, 59, 74, 200]
EDAD_LABELS = ["18-24", "25-34", "35-44", "45-59", "60-74", "75+"]
MIN_ELECCIONES = 5
MIN_LN = 30
LLAVE_CELDA = ["EDOCVE", "MPIOCVE", "SEXO", "grupo_edad"]

REGIONES_BANXICO = {
    "Norte": [2, 5, 8, 19, 26, 28],
    "Centro-Norte": [1, 3, 6, 10, 14, 16, 18, 24, 25, 32],
    "Centro": [9, 11, 13, 15, 17, 21, 22, 29],
    "Sur": [4, 7, 12, 20, 23, 27, 30, 31],
}
REGION = {edo: region for region, edos in REGIONES_BANXICO.items() for edo in edos}

CHIAPAS, MICHOACAN = 7, 16
CLAVES_PARACHO_PARACUARO = [65, 66]

VARIABLES_EIC = {
    "PCN_VPH_INTER": "internet", "PCN_VPH_PC": "computadora", "PCN_VPH_CEL": "celular",
    "PCN_VPH_TELEF": "telefono_fijo", "PCN_VPH_TVP": "tv_paga",
    "PCN_VPH_TV": "tv", "PCN_VPH_RADIO": "radio",
    "GRAPROES": "escolaridad", "PCN_PDESOCUP": "desocupacion",
    "PCN_POCUP_CPRO": "cuenta_propia", "PCN_POCUP_SPAG": "trabajo_sin_pago",
    "PCN_PSINDER": "sin_servicios_salud", "PCN_HOG_ALIM_N": "inseg_alimentaria",
    "PCN_HOG_GOB": "hog_programas_gob", "PCN_HOG_OPAIS": "hog_remesas",
    "PCN_VPH_PISOTI": "piso_tierra", "PCN_POB_IND": "pob_indigena",
    "PCN_P3YM_HLI": "habla_lengua_ind", "PCN_POB_AFRO": "pob_afro",
    "PCN_DESP_INSEG": "desplazamiento_inseg", "MEDIANA_POBTOT": "edad_mediana",
    "OCUPVIVPAR": "ocup_viv_2025", "POBTOT": "pob_2025",
}

VARIABLES_CONEVAL = {
    "Pobreza": "pobreza",
    "Pobreza extrema": "pobreza_extrema",
    "Carencia por acceso a la seguridad social": "carencia_seg_social",
    "Rezago educativo": "rezago_educativo",
    "Población con ingreso inferior a la línea de pobreza por ingresos": "ingreso_bajo_linea",
}

LOCALIDADES_AGRUPADAS_RURALES = ["9998", "9999"]
UMBRAL_LOCALIDAD_RURAL = 2500
MUNICIPIOS_ESCINDIDOS_TRAS_2020 = ["24028", "24059"]

INDICES_PCA = {
    "indice_conectividad": (["internet", "computadora", "celular", "telefono_fijo", "tv_paga"],
                            "internet"),
    "indice_socioeconomico": (["escolaridad", "indice_marginacion", "pobreza", "carencia_seg_social",
                               "inseg_alimentaria", "piso_tierra", "ocup_ingreso_menor_2sm"],
                              "escolaridad"),
}

log = Bitacora()


def limpiar_texto(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def cargar_ccpc():
    ccpc = pd.read_csv(PROC / "ccpc_mun_sexo_edad.csv.gz")
    log("== 1. CCPC ==")
    log(f"Filas cargadas: {len(ccpc):,}")

    nac = ccpc.groupby("AELEC")[["LN", "SV", "NV", "NS"]].sum()
    nac["part_SV_LN_%"] = (nac.SV / nac.LN * 100).round(1)
    nac["part_SV_SVNV_%"] = (nac.SV / (nac.SV + nac.NV) * 100).round(1)
    log("Participación nacional por elección (compárese con el tablero del INE):")
    log(nac.to_string())

    no_binaria = ccpc[ccpc.SEXO == 2].LN.sum()
    log(f"Lista nominal no binaria excluida (SEXO=2, solo 2024): {no_binaria:,}")
    return ccpc[ccpc.SEXO.isin([0, 1]) & (ccpc.EDAD >= 18)].copy()


def cargar_equivalencias():
    eq = pd.read_csv(PROC / "equivalencias_mun.csv", dtype={"CVEGEO_MUN": str})
    eq["norm"] = eq["nom_ine"].map(normalizar_nombre)
    return eq


def alinear_claves_chiapas(ccpc, eq):
    cambios = pd.read_csv(PROC / "catalogo_ine_cambios.csv")
    chis18 = cambios[(cambios.EDOCVE == CHIAPAS) & (cambios.AELEC == 2018) & (cambios.MPIOCVE >= 120)]
    remap = {}
    for _, r in chis18.iterrows():
        destino = eq[(eq.EDOCVE == CHIAPAS) & (eq["norm"] == r["norm"])]
        if len(destino) == 1:
            remap[int(r.MPIOCVE)] = int(destino.MPIOCVE.iloc[0])
    log("\n== 1a. Chiapas: reasignación de claves INE de 2018 a claves 2021/2024 ==")
    log(remap if remap else "Sin cambios")

    sel = (ccpc.AELEC == 2018) & (ccpc.EDOCVE == CHIAPAS) & ccpc.MPIOCVE.isin(list(remap))
    ccpc.loc[sel, "MPIOCVE"] = ccpc.loc[sel, "MPIOCVE"].map(remap)


def corregir_paracho_paracuaro(ccpc):
    a, b = CLAVES_PARACHO_PARACUARO
    ln = (ccpc[(ccpc.EDOCVE == MICHOACAN) & ccpc.MPIOCVE.isin([a, b])]
          .groupby(["AELEC", "MPIOCVE"]).LN.sum().unstack())
    log("\n== 1b. Michoacán claves 65/66, lista nominal por elección ==")
    log(ln.to_string())
    if not {2018, 2021, 2024}.issubset(ln.index):
        return

    referencia = ln.loc[[2018, 2024]].mean()
    v2021 = ln.loc[2021]
    distancia_igual = abs(v2021[a] - referencia[a]) + abs(v2021[b] - referencia[b])
    distancia_cruzada = abs(v2021[a] - referencia[b]) + abs(v2021[b] - referencia[a])
    if distancia_cruzada < distancia_igual:
        sel = (ccpc.AELEC == 2021) & (ccpc.EDOCVE == MICHOACAN) & ccpc.MPIOCVE.isin([a, b])
        ccpc.loc[sel, "MPIOCVE"] = ccpc.loc[sel, "MPIOCVE"].map({a: b, b: a})
        log("Decisión: en 2021 las CLAVES están invertidas -> se intercambian.")
    else:
        log("Decisión: las claves son consistentes; solo los nombres de 2021 estaban invertidos.")


def pendiente_por_fila(anios, Y):
    hay_dato = ~np.isnan(Y)
    X = np.broadcast_to(anios, Y.shape)
    x_media = np.where(hay_dato, X, 0).sum(axis=1) / hay_dato.sum(axis=1)
    y_media = np.nanmean(Y, axis=1)
    dx = np.where(hay_dato, X - x_media[:, None], 0)
    dy = np.where(hay_dato, Y - y_media[:, None], 0)
    return (dx * dy).sum(axis=1) / (dx ** 2).sum(axis=1)


def construir_celdas(ccpc):
    ccpc["grupo_edad"] = pd.cut(ccpc.EDAD, bins=EDAD_BINS, labels=EDAD_LABELS)
    agg = (ccpc.groupby(LLAVE_CELDA + ["AELEC"], observed=True)[["LN", "SV", "NV"]]
               .sum().reset_index())
    agg["part"] = agg.SV / (agg.SV + agg.NV) * 100
    agg.to_csv(PROC / "participacion_largo.csv", index=False)

    part = agg.pivot_table(index=LLAVE_CELDA, columns="AELEC", values="part", observed=True)
    ln = agg.pivot_table(index=LLAVE_CELDA, columns="AELEC", values="LN", observed=True)
    anios = sorted(part.columns)
    Y = part[anios].to_numpy()

    feat = pd.DataFrame(index=part.index)
    feat["n_elecciones"] = (~np.isnan(Y)).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feat["part_media"] = np.nanmean(Y, axis=1)
        feat["tendencia_pp_anio"] = pendiente_por_fila(np.array(anios, dtype=float), Y)
        feat["volatilidad"] = np.nanstd(Y, axis=1, ddof=1)
    feat["brecha_pres_int"] = (part[[a for a in PRESIDENCIALES if a in part]].mean(axis=1)
                               - part[[a for a in INTERMEDIAS if a in part]].mean(axis=1))
    for a in anios:
        feat[f"part_{a}"] = part[a]
    feat["ln_2024"] = ln.get(2024)
    feat["ln_min"] = ln.min(axis=1)
    feat = feat.reset_index()

    log("\n== 2. Celdas municipio x sexo x grupo de edad ==")
    log(f"Celdas totales: {len(feat):,}")
    suficientes = feat.n_elecciones >= MIN_ELECCIONES
    fuera_n = (~suficientes).sum()
    fuera_ln = (suficientes & (feat.ln_min < MIN_LN)).sum()
    feat = feat[suficientes & (feat.ln_min >= MIN_LN)].copy()
    log(f"Excluidas por tener menos de {MIN_ELECCIONES} elecciones: {fuera_n:,}")
    log(f"Excluidas por lista nominal menor a {MIN_LN} en alguna elección: {fuera_ln:,}")
    log(f"Celdas que quedan: {len(feat):,} "
        f"(representan {feat.ln_2024.sum():,.0f} personas en la LN 2024)")
    return feat


def asignar_clave_inegi(feat, eq):
    correcciones = pd.read_csv(MANUAL / "correcciones_equivalencias.csv", dtype={"CVEGEO_MUN": str})
    for _, r in correcciones.iterrows():
        sel = (eq.EDOCVE == r.EDOCVE) & (eq.MPIOCVE == r.MPIOCVE)
        eq.loc[sel, "CVEGEO_MUN"] = r.CVEGEO_MUN
    eq["CVEGEO_MUN"] = eq["CVEGEO_MUN"].str.zfill(5)
    duplicadas = eq[eq.CVEGEO_MUN.notna() & eq.duplicated("CVEGEO_MUN", keep=False)]

    log("\n== 3. Equivalencias INE -> INEGI ==")
    log(f"Correcciones manuales aplicadas: {len(correcciones)} | sin clave INEGI: "
        f"{eq.CVEGEO_MUN.isna().sum()} | claves INEGI duplicadas: {len(duplicadas)}")
    return feat.merge(eq[["EDOCVE", "MPIOCVE", "CVEGEO_MUN", "nom_ine"]],
                      on=["EDOCVE", "MPIOCVE"], how="left")


def contexto_intercensal():
    eic = pd.read_csv(PROC / "eic2025_mun.csv", dtype=str)
    ctx = eic[["CVEGEO_MUN", "NOM_ENT", "NOM_MUN"]].copy()
    for original, nuevo in VARIABLES_EIC.items():
        ctx[nuevo] = a_numero(eic[original])
    ctx["traslado_mas_1h"] = a_numero(eic["PCN_POCUP_2HOR"]) + a_numero(eic["PCN_POCUP_2HYM"])
    ctx["llego_de_fuera_5a"] = (a_numero(eic["PCN_PRESOE20"]) + a_numero(eic["PCN_PRESEUA20"])
                                + a_numero(eic["PCN_PRESOP20"]))
    ctx["CVEGEO_MUN"] = ctx["CVEGEO_MUN"].str.zfill(5)
    log(f"Intercensal 2025: {len(ctx):,} municipios")
    return ctx


def agregar_marginacion(ctx):
    imm = pd.read_csv(PROC / "excel" / "IMM_2020__IMM_2020.csv", dtype=str)
    imm = imm.rename(columns={"CVE_MUN": "CVEGEO_MUN"})
    imm["CVEGEO_MUN"] = imm["CVEGEO_MUN"].str.zfill(5)
    salida = imm[["CVEGEO_MUN", "GM_2020"]].copy()
    salida["indice_marginacion"] = a_numero(imm["IM_2020"])
    salida["ocup_ingreso_menor_2sm"] = a_numero(imm["PO2SM"])
    salida["pob_loc_menor_5000"] = a_numero(imm["PL.5000"])
    ctx = ctx.merge(salida, on="CVEGEO_MUN", how="left")
    log(f"IMM 2020 sin empate: {ctx.indice_marginacion.isna().sum()} municipios")
    return ctx


def agregar_pobreza(ctx):
    raw = pd.read_csv(PROC / "excel" / "Concentrado_indicadores_de_pobreza_2020__Concentrado_municipal.csv",
                      header=None, dtype=str)
    es_encabezado = raw.apply(lambda r: r.astype(str).str.strip().eq("Clave de municipio").any(), axis=1)
    fila_h = raw.index[es_encabezado][0]
    encabezado = raw.iloc[fila_h].astype(str).str.strip()
    grupos = raw.iloc[fila_h].ffill().fillna("").map(limpiar_texto)
    subtitulos = raw.iloc[fila_h + 1].fillna("").map(limpiar_texto)
    j_ent = grupos[encabezado == "Clave de entidad"].index[0]
    j_mun = grupos[encabezado == "Clave de municipio"].index[0]

    def columna_porcentaje_2020(etiqueta):
        js = [j for j in raw.columns if grupos[j] == etiqueta and subtitulos[j] == "Porcentaje 2020"]
        if len(js) != 1:
            raise ValueError(f"No se encontró '{etiqueta}' / 'Porcentaje 2020' en CONEVAL: {js}")
        return js[0]

    datos = raw.iloc[fila_h + 2:]
    datos = datos[datos[j_mun].astype(str).str.strip().str.fullmatch(r"\d+")]
    mun = datos[j_mun].astype(str).str.strip()
    ent = datos[j_ent].astype(str).str.strip()
    cve = np.where(mun.str.len() <= 3, ent.str.zfill(2) + mun.str.zfill(3), mun.str.zfill(5))

    coneval = pd.DataFrame({"CVEGEO_MUN": cve})
    for etiqueta, nuevo in VARIABLES_CONEVAL.items():
        coneval[nuevo] = a_numero(datos[columna_porcentaje_2020(etiqueta)].values)
    ctx = ctx.merge(coneval, on="CVEGEO_MUN", how="left")
    log(f"CONEVAL 2020: {len(coneval):,} municipios leídos, sin empate: {ctx.pobreza.isna().sum()}")
    return ctx


def agregar_ruralidad(ctx):
    ruta = next(RAW.rglob("conjunto_de_datos_iter_00*.csv"))
    it = leer_csv(ruta, usecols=["ENTIDAD", "MUN", "LOC", "POBTOT", "OCUPVIVPAR"], dtype=str)
    it = it[(it.ENTIDAD != "00") & (it.MUN != "000")].copy()
    it["POBTOT"] = a_numero(it.POBTOT)
    it["OCUPVIVPAR"] = a_numero(it.OCUPVIVPAR)
    it["CVEGEO_MUN"] = it.ENTIDAD.str.zfill(2) + it.MUN.str.zfill(3)

    totales = it[it.LOC == "0000"][["CVEGEO_MUN", "POBTOT", "OCUPVIVPAR"]].rename(
        columns={"POBTOT": "pob_2020", "OCUPVIVPAR": "ocup_viv_2020"})
    locs = it[it.LOC != "0000"]
    agrupadas = locs.LOC.isin(LOCALIDADES_AGRUPADAS_RURALES)
    es_rural = (locs.POBTOT < UMBRAL_LOCALIDAD_RURAL) | agrupadas
    rural = locs[es_rural].groupby("CVEGEO_MUN").POBTOT.sum().rename("pob_rural_2020")
    n_loc = locs[~agrupadas].groupby("CVEGEO_MUN").size().rename("n_localidades")
    totales = (totales.merge(rural, on="CVEGEO_MUN", how="left")
                      .merge(n_loc, on="CVEGEO_MUN", how="left"))
    totales["pct_pob_rural"] = totales.pob_rural_2020.fillna(0) / totales.pob_2020 * 100

    ctx = ctx.merge(totales[["CVEGEO_MUN", "pct_pob_rural", "n_localidades", "ocup_viv_2020"]],
                    on="CVEGEO_MUN", how="left")
    ctx["cambio_pob_20_25"] = (ctx.ocup_viv_2025 / ctx.ocup_viv_2020 - 1) * 100
    ctx.loc[ctx.CVEGEO_MUN.isin(MUNICIPIOS_ESCINDIDOS_TRAS_2020), "cambio_pob_20_25"] = np.nan
    log(f"ITER 2020 sin empate: {ctx.pct_pob_rural.isna().sum()} municipios")
    return ctx


def agregar_indices_pca(ctx):
    for nombre, (cols, ancla) in INDICES_PCA.items():
        Z = ctx[cols].copy()
        imputados = Z.isna().any(axis=1).sum()
        Z = Z.fillna(Z.median())
        Zs = StandardScaler().fit_transform(Z)
        pca = PCA(n_components=1).fit(Zs)
        score = pca.transform(Zs)[:, 0]
        cargas = pca.components_[0]
        if np.corrcoef(score, Z[ancla])[0, 1] < 0:
            score, cargas = -score, -cargas
        ctx[nombre] = score
        log(f"{nombre}: varianza explicada {pca.explained_variance_ratio_[0]:.1%}; "
            f"cargas {dict(zip(cols, np.round(cargas, 2)))}; imputados con mediana: {imputados}")
    return ctx


def construir_contexto():
    log("\n== 4. Contexto municipal ==")
    ctx = contexto_intercensal()
    ctx = agregar_marginacion(ctx)
    ctx = agregar_pobreza(ctx)
    ctx = agregar_ruralidad(ctx)
    ctx = agregar_indices_pca(ctx)
    ctx["region"] = ctx.CVEGEO_MUN.str[:2].astype(int).map(REGION)
    ctx.to_csv(PROC / "contexto_mun.csv", index=False)
    return ctx


def composicion_secciones_2024():
    tipo = pd.read_csv(PROC / "ccpc_mun_tiposec.csv")
    tipo = tipo[tipo.AELEC == 2024].pivot_table(index=["EDOCVE", "MPIOCVE"], columns="TIPOSEC",
                                                values="LN", aggfunc="sum").fillna(0)
    return (tipo.div(tipo.sum(axis=1), axis=0) * 100).add_prefix("pct_ln_secc_").reset_index()


def construir_base(feat, ctx):
    base = (feat.merge(composicion_secciones_2024(), on=["EDOCVE", "MPIOCVE"], how="left")
                .merge(ctx, on="CVEGEO_MUN", how="left"))
    base["region"] = base.EDOCVE.map(REGION)
    base["sexo"] = base.SEXO.map({0: "Hombres", 1: "Mujeres"})
    base.to_csv(PROC / "base_analisis.csv", index=False)

    log("\n== 5. Base final ==")
    log(f"Filas: {len(base):,} | municipios: {base.CVEGEO_MUN.nunique():,}")
    log(f"Filas sin contexto Intercensal: {base.internet.isna().sum():,}")
    log("Participación media por grupo de edad y sexo (%):")
    log(base.pivot_table(index="grupo_edad", columns="sexo", values="part_media",
                         aggfunc="mean", observed=True).round(1).to_string())
    log("Participación media por región (%):")
    log(base.groupby("region").part_media.mean().round(1).to_string())


def main():
    ccpc = cargar_ccpc()
    eq = cargar_equivalencias()
    alinear_claves_chiapas(ccpc, eq)
    corregir_paracho_paracuaro(ccpc)
    feat = construir_celdas(ccpc)
    feat = asignar_clave_inegi(feat, eq)
    ctx = construir_contexto()
    construir_base(feat, ctx)

    log.guardar(PROC / "validacion_paso3.txt")
    print("\nListo. Bitácora en data/processed/validacion_paso3.txt")


if __name__ == "__main__":
    main()
