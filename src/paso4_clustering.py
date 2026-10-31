"""
Paso 4 · Clustering de perfiles de participación e interpretación.

Entradas (data/processed/)
    base_analisis.csv, participacion_largo.csv

Salidas
    data/processed/clustering/   tablas de selección de k, perfiles y municipios
    outputs/figuras/             gráficas
    data/processed/clustering/bitacora_paso4.txt

Notas metodológicas
    - Las variables de participación pesan 1 y las de contexto 0.5 en la matriz de clustering.
    - indice_socioeconomico queda fuera del clustering por su correlación con
      indice_conectividad (r = 0.90 > 0.85); se usa solo para caracterizar perfiles.
    - Se evalúan k = 4..7, pero solo se elige entre 4 y 6 (requisito de la convocatoria).
    - Cada celda pondera por su lista nominal 2024.

Uso (desde la raíz del proyecto)
    python src/paso4_clustering.py            # elige k automáticamente entre 4 y 6
    python src/paso4_clustering.py --k 5      # fuerza un valor de k
"""
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from comun import FIGURAS, PROC, Bitacora

matplotlib.use("Agg")

OUT = PROC / "clustering"

SEED = 42
K_EVALUADOS = range(4, 8)
K_PERMITIDOS = [4, 5, 6]
TAMANO_MIN_CLUSTER_PCT = 3
UMBRAL_CORRELACION = 0.85
TAMANO_MUESTRA_SILUETA = 12000

VARS_PART = ["part_media", "tendencia_pp_anio", "volatilidad", "brecha_pres_int"]
VARS_CTX = ["indice_conectividad", "pct_pob_rural"]
PESO_CTX = 0.5
VARS_CLUSTER = VARS_PART + VARS_CTX
VARS_DESC = VARS_CLUSTER + [
    "indice_socioeconomico", "escolaridad", "desocupacion", "cuenta_propia", "traslado_mas_1h", "pobreza",
    "indice_marginacion", "internet", "celular", "computadora", "radio", "tv",
    "pob_indigena", "habla_lengua_ind", "pob_afro", "llego_de_fuera_5a",
    "desplazamiento_inseg", "hog_remesas", "hog_programas_gob", "cambio_pob_20_25",
    "part_2009", "part_2012", "part_2015", "part_2018", "part_2021", "part_2024"]
VARS_COMPOSICION = ["sexo", "grupo_edad", "region"]
LLAVE_CELDA = ["EDOCVE", "MPIOCVE", "SEXO", "grupo_edad"]

log = Bitacora()


def promedio_ponderado(df, cols, pesos):
    def media(c):
        validos = c.notna()
        return np.average(c[validos], weights=pesos[validos]) if validos.any() else np.nan
    return df[cols].apply(media)


def preparar_matriz(base):
    log(f"Celdas: {len(base):,}")
    X = base[VARS_CLUSTER].copy()
    faltan = X.isna().sum()
    log("Valores faltantes imputados con la mediana:\n" + faltan[faltan > 0].to_string())
    X = X.fillna(X.median())

    corr = X.corr().round(2)
    log("\nCorrelaciones entre variables de clustering:\n" + corr.to_string())
    altas = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr) for b in corr.columns[i + 1:]
             if abs(corr.loc[a, b]) > UMBRAL_CORRELACION]
    log(f"Pares con |r| > {UMBRAL_CORRELACION}: {altas if altas else 'ninguno'}")

    Z = StandardScaler().fit_transform(X)
    Z[:, len(VARS_PART):] *= PESO_CTX
    peso = base["ln_2024"].fillna(base["ln_min"]).to_numpy()
    return X, Z, peso


def estabilidad(Z, peso, k, etiquetas):
    ari_semillas = [
        adjusted_rand_score(etiquetas, KMeans(n_clusters=k, n_init=5, random_state=s)
                            .fit(Z, sample_weight=peso).labels_)
        for s in range(1, 11)]
    ari_submuestras = []
    for s in range(10):
        idx = np.random.default_rng(100 + s).choice(len(Z), size=int(0.8 * len(Z)), replace=False)
        sub = KMeans(n_clusters=k, n_init=5, random_state=s).fit(Z[idx], sample_weight=peso[idx])
        ari_submuestras.append(adjusted_rand_score(etiquetas[idx], sub.labels_))
    return np.mean(ari_semillas), np.mean(ari_submuestras)


def evaluar_k(Z, peso):
    rng = np.random.default_rng(SEED)
    muestra = rng.choice(len(Z), size=min(TAMANO_MUESTRA_SILUETA, len(Z)), replace=False)
    filas, modelos = [], {}
    for k in K_EVALUADOS:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Z, sample_weight=peso)
        modelos[k] = km
        etiquetas = km.labels_
        ari_semillas, ari_submuestras = estabilidad(Z, peso, k, etiquetas)
        filas.append({
            "k": k,
            "inercia": km.inertia_,
            "silueta": silhouette_score(Z[muestra], etiquetas[muestra]),
            "calinski_harabasz": calinski_harabasz_score(Z, etiquetas),
            "davies_bouldin": davies_bouldin_score(Z, etiquetas),
            "estabilidad_semillas_ARI": ari_semillas,
            "estabilidad_submuestras_ARI": ari_submuestras,
            "tamano_min_%": np.bincount(etiquetas, weights=peso).min() / peso.sum() * 100,
        })
    seleccion = pd.DataFrame(filas).round(3)
    seleccion.to_csv(OUT / "seleccion_k.csv", index=False)
    log("\nSelección de k:\n" + seleccion.to_string(index=False))
    return seleccion, modelos


def elegir_k(seleccion, k_forzado):
    if k_forzado:
        log(f"k forzado por el usuario: {k_forzado}")
        return k_forzado
    candidatos = seleccion[seleccion.k.isin(K_PERMITIDOS)
                           & (seleccion["tamano_min_%"] >= TAMANO_MIN_CLUSTER_PCT)]
    k = int(candidatos.sort_values(["silueta", "estabilidad_submuestras_ARI"], ascending=False).k.iloc[0])
    log(f"k elegido (mayor silueta entre 4 y 6, con clusters de al menos 3 % de la LN): {k}")
    return k


def graficar_seleccion_k(seleccion, k):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(seleccion.k, seleccion.inercia, "o-")
    ax[0].set_title("Método del codo")
    ax[1].plot(seleccion.k, seleccion.silueta, "o-", color="purple")
    ax[1].set_title("Coeficiente de silueta")
    for a in ax:
        a.set_xlabel("k")
        a.axvline(k, ls="--", color="gray")
    fig.tight_layout()
    fig.savefig(FIGURAS / "seleccion_k.png", dpi=200)
    plt.close(fig)


def asignar_perfiles(base, km, Z, k):
    orden = np.argsort(-km.cluster_centers_[:, 0])
    renumeracion = {anterior: nuevo + 1 for nuevo, anterior in enumerate(orden)}
    base["perfil"] = pd.Series(km.labels_).map(renumeracion).to_numpy()

    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=SEED).fit(Z)
    log(f"\nConcordancia K-means vs GMM (ARI): {adjusted_rand_score(base.perfil, gmm.predict(Z)):.3f}")


def resumir_perfiles(base):
    por_perfil = base.groupby("perfil")
    res = por_perfil.apply(lambda g: promedio_ponderado(g, VARS_DESC, g["ln_2024"].fillna(0)))
    res.insert(0, "lista_nominal_2024", por_perfil.ln_2024.sum())
    res.insert(1, "pct_lista_nominal", res.lista_nominal_2024 / res.lista_nominal_2024.sum() * 100)
    res.insert(2, "n_celdas", por_perfil.size())
    res.insert(3, "n_municipios", por_perfil.CVEGEO_MUN.nunique())
    nacional = promedio_ponderado(base, VARS_DESC, base["ln_2024"].fillna(0))
    res.loc["Nacional"] = ([res.lista_nominal_2024.sum(), 100, len(base), base.CVEGEO_MUN.nunique()]
                           + list(nacional))
    res.round(2).to_csv(OUT / "perfiles_resumen.csv")
    log("\nResumen por perfil (promedios ponderados por lista nominal):\n"
        + res[["pct_lista_nominal", "n_municipios"] + VARS_CLUSTER + ["escolaridad", "pob_indigena"]]
        .round(1).to_string())
    return res, nacional


def composicion(base, var):
    t = base.pivot_table(index="perfil", columns=var, values="ln_2024", aggfunc="sum",
                         observed=True).fillna(0)
    return t.div(t.sum(axis=1), axis=0) * 100


def composicion_perfiles(base):
    comp = pd.concat({"sexo": composicion(base, "sexo"), "edad": composicion(base, "grupo_edad"),
                      "region": composicion(base, "region")}, axis=1).round(1)
    comp.to_csv(OUT / "perfiles_composicion.csv")
    log("\nComposición de cada perfil (% de su lista nominal):\n" + comp.to_string())

    total = {v: base.groupby(v, observed=True).ln_2024.sum() / base.ln_2024.sum() * 100
             for v in VARS_COMPOSICION}
    indice = pd.concat({v: composicion(base, v).div(total[v], axis=1) * 100
                        for v in VARS_COMPOSICION}, axis=1).round(0)
    indice.to_csv(OUT / "perfiles_indice_sobrerrepresentacion.csv")


def participacion(df):
    return df.SV / (df.SV + df.NV) * 100


def trayectorias(base):
    larg = pd.read_csv(PROC / "participacion_largo.csv")
    larg = larg.merge(base[LLAVE_CELDA + ["perfil"]], on=LLAVE_CELDA, how="inner")
    tray = participacion(larg.groupby(["perfil", "AELEC"])[["SV", "NV"]].sum()).unstack().round(1)
    tray.loc["Nacional"] = participacion(larg.groupby("AELEC")[["SV", "NV"]].sum()).round(1)
    tray.to_csv(OUT / "perfiles_trayectoria.csv")
    log("\nParticipación por elección y perfil (%):\n" + tray.to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    for p in tray.index:
        es_nacional = p == "Nacional"
        ax.plot(tray.columns, tray.loc[p], "o-", lw=3 if es_nacional else 2,
                color="black" if es_nacional else None,
                label=p if es_nacional else f"Perfil {p}")
    ax.set_ylabel("% participación")
    ax.set_title("Participación por perfil, 2009-2024")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURAS / "trayectorias_perfiles.png", dpi=200)
    plt.close(fig)


def mapa_calor(base, res, nacional, k):
    zc = (res.loc[res.index != "Nacional", VARS_CLUSTER] - nacional[VARS_CLUSTER]) \
        / base[VARS_CLUSTER].std()
    fig, ax = plt.subplots(figsize=(9, 0.6 * k + 2))
    im = ax.imshow(zc.to_numpy(dtype=float), cmap="RdBu", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(zc.shape[1]))
    ax.set_xticklabels(zc.columns, rotation=40, ha="right")
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"Perfil {i}" for i in zc.index])
    for i in range(zc.shape[0]):
        for j in range(zc.shape[1]):
            ax.text(j, i, f"{zc.iat[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="desviaciones estándar vs. nacional")
    fig.tight_layout()
    fig.savefig(FIGURAS / "mapa_calor_perfiles.png", dpi=200)
    plt.close(fig)


def explicar_con_arbol(base, X, peso):
    Xa = pd.concat([X, pd.get_dummies(base[VARS_COMPOSICION].astype(str), dtype=int)], axis=1)
    arbol = DecisionTreeClassifier(max_depth=4, min_samples_leaf=200, random_state=SEED)
    arbol.fit(Xa, base.perfil, sample_weight=peso)
    exactitud = (arbol.predict(Xa) == base.perfil).mean()
    importancia = pd.Series(arbol.feature_importances_, index=Xa.columns).sort_values(ascending=False)
    reglas = export_text(arbol, feature_names=list(Xa.columns), decimals=2)
    (OUT / "arbol_reglas.txt").write_text(
        f"Exactitud del árbol (profundidad 4): {exactitud:.1%}\n\nImportancia de variables:\n"
        f"{importancia[importancia > 0].round(3).to_string()}\n\nReglas:\n{reglas}", encoding="utf-8")
    log(f"\nÁrbol de decisión: exactitud {exactitud:.1%}. Variables más importantes:\n"
        + importancia.head(8).round(3).to_string())


def municipios_por_perfil(base):
    mun = base.pivot_table(index=["CVEGEO_MUN", "NOM_ENT", "NOM_MUN"], columns="perfil",
                           values="ln_2024", aggfunc="sum").fillna(0)
    mun_pct = mun.div(mun.sum(axis=1), axis=0) * 100
    part_mun = base.groupby("CVEGEO_MUN").apply(
        lambda g: np.average(g.part_2024.fillna(g.part_media), weights=g.ln_2024.fillna(1)))

    dominante = pd.DataFrame({"perfil_dominante": mun_pct.idxmax(axis=1),
                              "pct_ln_perfil_dominante": mun_pct.max(axis=1).round(1),
                              "ln_2024": mun.sum(axis=1)}).reset_index()
    dominante["part_2024_mun"] = dominante.CVEGEO_MUN.map(part_mun).round(1)
    dominante.to_csv(OUT / "municipios_perfil_dominante.csv", index=False)

    prioritarios = []
    for p in sorted(base.perfil.unique()):
        t = mun[p].rename("ln_perfil").reset_index()
        t["pct_ln_mun_en_perfil"] = mun_pct[p].values.round(1)
        t["part_2024_mun"] = t.CVEGEO_MUN.map(part_mun).round(1)
        t = t.sort_values("ln_perfil", ascending=False).head(10)
        t.insert(0, "perfil", p)
        prioritarios.append(t)
    pd.concat(prioritarios).to_csv(OUT / "municipios_prioritarios_por_perfil.csv", index=False)
    log("\nMunicipios por perfil dominante:\n"
        + dominante.perfil_dominante.value_counts().sort_index().to_string())


def main():
    parser = argparse.ArgumentParser(description="Clustering de perfiles de participación electoral.")
    parser.add_argument("--k", type=int, default=None, help="número de perfiles a forzar")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(PROC / "base_analisis.csv", dtype={"CVEGEO_MUN": str})
    X, Z, peso = preparar_matriz(base)

    seleccion, modelos = evaluar_k(Z, peso)
    k = elegir_k(seleccion, args.k)
    graficar_seleccion_k(seleccion, k)

    asignar_perfiles(base, modelos[k], Z, k)
    res, nacional = resumir_perfiles(base)
    composicion_perfiles(base)
    trayectorias(base)
    mapa_calor(base, res, nacional, k)
    explicar_con_arbol(base, X, peso)
    municipios_por_perfil(base)

    base.to_csv(OUT / "celdas_con_perfil.csv", index=False)
    log.guardar(OUT / "bitacora_paso4.txt")
    print("\nListo. Revisa data/processed/clustering/ y outputs/figuras/")


if __name__ == "__main__":
    main()
