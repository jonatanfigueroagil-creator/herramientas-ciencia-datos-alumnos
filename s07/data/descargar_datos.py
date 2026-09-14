# -*- coding: utf-8 -*-
"""
Descarga y verificacion de los datos de la Sesion S07
(Clustering jerarquico + DBSCAN + deteccion de anomalias) del curso UPC
"Herramientas para la Ciencia de Datos".

TRES conjuntos:

  1) MAMMOGRAPHY (replicacion de anomalias) -> "mammography.csv"  11183 x 7
     Benchmark de ODDS (Outlier Detection DataSets, Stony Brook). Es el mismo
     dataset con el que Liu, Ting & Zhou (2008) evaluan Isolation Forest.
     El sitio original http://odds.cs.stonybrook.edu/ tiene el certificado SSL
     roto (verificado 18/07/2026) -> se resuelve por OpenML.
       - Carga: sklearn.datasets.fetch_openml(data_id=310)  (name="mammography")
       - 11183 filas, 6 features numericas (attr1..attr6) + etiqueta binaria.
       - Etiqueta: 'class' en {-1 normal, 1 anomalia}; 260 anomalias = 2.32%.
     Cita del paper: Liu, F.T., Ting, K.M. & Zhou, Z.-H. (2008). "Isolation
     Forest". Proc. IEEE ICDM 2008. Origen del dataset: Woods et al. (1993),
     mamografias digitalizadas; calcificaciones = clase minoritaria (anomalia).

  2) SHUTTLE-ODDS (replicacion de anomalias, 2.o dataset) -> "shuttle_odds.csv"
     49097 x 10. Statlog (Shuttle) de UCI, con el preprocesamiento CANONICO de
     ODDS para deteccion de anomalias:
       - Carga: sklearn.datasets.fetch_openml(data_id=40685)  (58000 x 9, 7 clases)
       - Se DESCARTA la clase 4; la clase 1 es el inlier (normal) y las clases
         pequenas {2,3,5,6,7} se combinan como anomalias.
       - Resultado: 49097 filas, 9 features (A1..A9) + etiqueta binaria;
         3511 anomalias = 7.15%.
     Mismo paper de la replica (Liu et al. 2008), que reporta AUC de IF ~0.99
     en Shuttle. LOF (Breunig et al. 2000) sirve de contraste.

  3) ONLINE RETAIL II (negocio / laboratorio RFM) -> "online_retail_II_muestra.csv"
     El COMPLETO son ~1.07M filas en un .xlsx de ~45.6 MB (dos hojas: "Year
     2009-2010" y "Year 2010-2011") -> NUNCA se guarda en OneDrive. Se descarga
     el .zip de UCI SOLO en runtime (a un directorio temporal del sistema, fuera
     de OneDrive), se limpia y se guarda una MUESTRA <=1 MB por cliente.
       - Fuente: UCI id 502, https://archive.ics.uci.edu/dataset/502/online+retail+ii
         zip directo: https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
         SHA256(zip) = 572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb
       - Limpieza (documentada en diccionario_datos.md): quitar faltantes de
         Customer ID, cancelaciones (Invoice que empieza por 'C'), Quantity<=0 y
         Price<=0. 1.067.371 -> 805.549 filas limpias.
       - Muestra: 140 clientes elegidos al azar (RandomState=42) con TODA su
         historia (Recency/Frequency/Monetary quedan exactas por cliente).
         Columnas esenciales de RFM: Invoice, InvoiceDate, Quantity, Price,
         Customer ID, Country.
     Cita: Chen, D. (2019). Online Retail II. UCI Machine Learning Repository.
     https://doi.org/10.24432/C5CG6D . Licencia CC BY 4.0.

Idempotente: si un archivo ya existe y su esquema (filas x columnas + columnas
clave) es correcto, NO se vuelve a generar. Funciona en local (Windows/venv) y
en Google Colab. fetch_openml cachea en ~/scikit_learn_data (fuera de OneDrive);
el .xlsx completo de Online Retail II se descarga a un temporal y NO se conserva
en la carpeta del curso.

Uso:
    "<python_del_venv>" descargar_datos.py            # genera + verifica esquema
    "<python_del_venv>" descargar_datos.py --smoke     # ademas corre el smoke de viabilidad
    "<python_del_venv>" descargar_datos.py --full-retail  # baja+limpia el completo (no lo guarda)
    # o en Colab:  %run descargar_datos.py

Fecha de verificacion de fuentes: 18/07/2026.
"""
import hashlib
import io
import os
import sys
import tempfile
import zipfile
import warnings

warnings.filterwarnings("ignore")

try:
    import requests
except ImportError:  # en Colab requests viene preinstalado
    requests = None

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (curso UPC S07 descargar_datos)"}

if "google.colab" in sys.modules:
    DATA_DIR = os.getcwd()
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Cache del .xlsx completo FUERA de OneDrive (temporal del sistema).
CACHE_DIR = os.path.join(tempfile.gettempdir(), "curso_upc_s07_online_retail")

MAMMO_FILE = os.path.join(DATA_DIR, "mammography.csv")            # 11183 x 7
SHUTTLE_FILE = os.path.join(DATA_DIR, "shuttle_odds.csv")         # 49097 x 10
RETAIL_SAMPLE = os.path.join(DATA_DIR, "online_retail_II_muestra.csv")

# --- OpenML data_ids (fallback de ODDS, verificados 18/07/2026) ---
MAMMO_ID = 310       # mammography  11183 x (6 feat + class)
SHUTTLE_ID = 40685   # statlog shuttle 58000 x (9 feat + class 7 clases)

# --- Online Retail II (UCI 502) ---
RETAIL_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
)
RETAIL_ZIP_SHA256 = (
    "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"
)
RETAIL_N_CLIENTES = 140         # -> muestra <=1 MB, RandomState=42
RETAIL_RANDOM_STATE = 42
RETAIL_COLS = ["Invoice", "InvoiceDate", "Quantity", "Price",
               "Customer ID", "Country"]

# Esquemas esperados
MAMMO_FEATS = [f"attr{i}" for i in range(1, 7)]
SHUTTLE_FEATS = [f"A{i}" for i in range(1, 10)]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str) -> bytes:
    if requests is None:
        raise RuntimeError("requests no disponible")
    r = requests.get(url, headers=HEADERS, timeout=300)
    r.raise_for_status()
    return r.content


def _csv_valido(path, key_cols, nrows) -> bool:
    """Idempotencia: CSV presente y con esquema correcto."""
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path, nrows=5)
        faltan = [c for c in key_cols if c not in df.columns]
        if faltan:
            print(f"  ! {os.path.basename(path)}: faltan columnas {faltan}; se regenera.")
            return False
        if nrows is not None:
            n = sum(1 for _ in open(path, encoding="utf-8")) - 1
            if n != nrows:
                print(f"  ! {os.path.basename(path)}: {n} filas != {nrows}; se regenera.")
                return False
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ! {os.path.basename(path)} no valida ({e}); se regenera.")
        return False


# ---------------------------------------------------------------------------
# 1) MAMMOGRAPHY (replicacion Isolation Forest)
# ---------------------------------------------------------------------------
def obtener_mammography() -> pd.DataFrame:
    print("== MAMMOGRAPHY 11183x7 (ODDS; replica Liu et al. 2008; via OpenML 310) ==")
    if _csv_valido(MAMMO_FILE, MAMMO_FEATS + ["anomalia"], 11183):
        print(f"  = presente y verificado: {MAMMO_FILE}")
        return pd.read_csv(MAMMO_FILE)
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=MAMMO_ID, as_frame=True)
    df = d.data.copy()
    df.columns = MAMMO_FEATS
    y = d.target.astype(int)
    df["anomalia"] = (y == 1).astype(int).values   # 1 = calcificacion (anomalia)
    if df.shape != (11183, 7):
        raise ValueError(f"esquema inesperado {df.shape} != (11183, 7)")
    tasa = df["anomalia"].mean() * 100
    if not (2.0 < tasa < 2.7):
        raise ValueError(f"tasa de anomalia {tasa:.2f}% fuera de lo esperado (~2.32%)")
    df.to_csv(MAMMO_FILE, index=False, encoding="utf-8")
    print(f"  + cargado via fetch_openml(data_id={MAMMO_ID}); anomalia={tasa:.2f}% "
          f"-> {MAMMO_FILE}")
    return df


# ---------------------------------------------------------------------------
# 2) SHUTTLE-ODDS (replicacion Isolation Forest, 2.o dataset)
# ---------------------------------------------------------------------------
def obtener_shuttle() -> pd.DataFrame:
    print("== SHUTTLE-ODDS 49097x10 (ODDS; replica Liu et al. 2008; via OpenML 40685) ==")
    if _csv_valido(SHUTTLE_FILE, SHUTTLE_FEATS + ["anomalia"], 49097):
        print(f"  = presente y verificado: {SHUTTLE_FILE}")
        return pd.read_csv(SHUTTLE_FILE)
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=SHUTTLE_ID, as_frame=True)
    X = d.data.copy()
    X.columns = SHUTTLE_FEATS
    y = d.target.astype(int)
    # Preprocesamiento canonico ODDS: descartar clase 4; clase 1 = normal;
    # clases {2,3,5,6,7} = anomalias.
    mask = (y != 4).values
    df = X.loc[mask].reset_index(drop=True)
    yb = y[mask].isin([2, 3, 5, 6, 7]).astype(int).reset_index(drop=True)
    df["anomalia"] = yb.values
    if df.shape != (49097, 10):
        raise ValueError(f"esquema inesperado {df.shape} != (49097, 10)")
    tasa = df["anomalia"].mean() * 100
    if not (6.5 < tasa < 7.8):
        raise ValueError(f"tasa de anomalia {tasa:.2f}% fuera de lo esperado (~7.15%)")
    df.to_csv(SHUTTLE_FILE, index=False, encoding="utf-8")
    print(f"  + cargado via fetch_openml(data_id={SHUTTLE_ID}) + preproc ODDS; "
          f"anomalia={tasa:.2f}% -> {SHUTTLE_FILE}")
    return df


# ---------------------------------------------------------------------------
# 3) ONLINE RETAIL II (negocio RFM) — completo NUNCA en OneDrive
# ---------------------------------------------------------------------------
def _descargar_retail_zip() -> bytes:
    """Descarga (o reutiliza el cache temporal) del .zip de UCI. Fuera de OneDrive."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    zp = os.path.join(CACHE_DIR, "online_retail_II.zip")
    if os.path.exists(zp) and sha256_file(zp) == RETAIL_ZIP_SHA256:
        print(f"  = zip en cache temporal ({zp})")
        return open(zp, "rb").read()
    print(f"  ... descargando {RETAIL_ZIP_URL} (~45.6 MB, solo runtime)")
    raw = _fetch(RETAIL_ZIP_URL)
    sha = sha256_bytes(raw)
    aviso = "" if sha == RETAIL_ZIP_SHA256 else "  [SHA256 de fuente NUEVO]"
    open(zp, "wb").write(raw)
    print(f"  ... zip -> {zp}  SHA256={sha[:16]}...{aviso}")
    return raw


def cargar_retail_completo_limpio() -> pd.DataFrame:
    """Devuelve el DataFrame COMPLETO limpio (~805.549 filas). No lo guarda en OneDrive."""
    raw = _descargar_retail_zip()
    z = zipfile.ZipFile(io.BytesIO(raw))
    name = [n for n in z.namelist() if n.lower().endswith(".xlsx")][0]
    print(f"  ... leyendo {name} (dos hojas, ~1.07M filas; tarda ~1-2 min)")
    sheets = pd.read_excel(io.BytesIO(z.read(name)), sheet_name=None, engine="openpyxl")
    df = pd.concat(sheets.values(), ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    n0 = len(df)
    df = df[df["Customer ID"].notna()]
    df = df[~df["Invoice"].astype(str).str.startswith("C")]   # cancelaciones
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]         # devoluciones / precios nulos
    df["Customer ID"] = df["Customer ID"].astype(int)
    print(f"  ... limpieza: {n0} -> {len(df)} filas  "
          f"({df['Customer ID'].nunique()} clientes, {df['Country'].nunique()} paises)")
    return df.reset_index(drop=True)


def obtener_retail_muestra() -> pd.DataFrame:
    print("== ONLINE RETAIL II — MUESTRA (negocio RFM; UCI 502; completo NUNCA en OneDrive) ==")
    if _csv_valido(RETAIL_SAMPLE, RETAIL_COLS, None):
        # validar que la muestra tenga el nº de clientes esperado
        df = pd.read_csv(RETAIL_SAMPLE, parse_dates=["InvoiceDate"])
        if df["Customer ID"].nunique() == RETAIL_N_CLIENTES:
            print(f"  = presente y verificado: {RETAIL_SAMPLE} "
                  f"({len(df)} filas, {RETAIL_N_CLIENTES} clientes)")
            return df
        print("  ! nº de clientes distinto al esperado; se regenera la muestra.")

    full = cargar_retail_completo_limpio()
    custs = np.sort(full["Customer ID"].unique())
    rng = np.random.RandomState(RETAIL_RANDOM_STATE)
    sel = rng.choice(custs, size=RETAIL_N_CLIENTES, replace=False)
    muestra = (full[full["Customer ID"].isin(sel)][RETAIL_COLS]
               .sort_values("InvoiceDate").reset_index(drop=True))
    muestra.to_csv(RETAIL_SAMPLE, index=False, encoding="utf-8")
    mb = os.path.getsize(RETAIL_SAMPLE) / 1e6
    print(f"  + muestra {RETAIL_N_CLIENTES} clientes (rs={RETAIL_RANDOM_STATE}) -> "
          f"{RETAIL_SAMPLE} ({len(muestra)} filas, {mb:.3f} MB)")
    if mb > 1.0:
        print(f"  ! AVISO: la muestra pesa {mb:.3f} MB (>1 MB). Reducir "
              f"RETAIL_N_CLIENTES.")
    return muestra


# ---------------------------------------------------------------------------
# SMOKE de VIABILIDAD (reporta valores OBSERVADOS; NO define targets)
# ---------------------------------------------------------------------------
def _auc_if_lof(X, y):
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    s_if = -IsolationForest(n_estimators=100, random_state=42,
                            contamination="auto").fit(Xs).score_samples(Xs)
    auc_if = roc_auc_score(y, s_if)
    lof = LocalOutlierFactor(n_neighbors=20, contamination="auto")
    lof.fit_predict(Xs)
    auc_lof = roc_auc_score(y, -lof.negative_outlier_factor_)
    return auc_if, auc_lof


def smoke():
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import AgglomerativeClustering, DBSCAN
    from sklearn.metrics import silhouette_score

    print("\n########## SMOKE DE VIABILIDAD S07 ##########")

    # -- (a) Anomalias: AUC-ROC de IsolationForest vs LOF --
    print("\n== (a) ANOMALIAS: AUC-ROC Isolation Forest vs LOF (etiqueta real) ==")
    mam = obtener_mammography()
    Xm = mam[MAMMO_FEATS].values
    ym = mam["anomalia"].values
    a_if, a_lof = _auc_if_lof(Xm, ym)
    print(f"  Mammography  n={len(ym)}  anom={ym.mean()*100:.2f}%  "
          f"AUC IF={a_if:.4f}  AUC LOF={a_lof:.4f}")
    sh = obtener_shuttle()
    Xs = sh[SHUTTLE_FEATS].values
    ys = sh["anomalia"].values
    a_if2, a_lof2 = _auc_if_lof(Xs, ys)
    print(f"  Shuttle-ODDS n={len(ys)}  anom={ys.mean()*100:.2f}%  "
          f"AUC IF={a_if2:.4f}  AUC LOF={a_lof2:.4f}")

    # -- (b) Clustering sobre una muestra: Ward + DBSCAN + silhouette --
    print("\n== (b) CLUSTERING (Ward + DBSCAN + silhouette) sobre muestra de 1500 ==")
    rng = np.random.RandomState(42)
    idx = rng.choice(len(Xm), size=1500, replace=False)
    Xsamp = StandardScaler().fit_transform(Xm[idx])
    ward = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(Xsamp)
    print(f"  Ward k=3  silhouette={silhouette_score(Xsamp, ward):.4f}  "
          f"tam={np.bincount(ward)}")
    db = DBSCAN(eps=0.6, min_samples=10).fit_predict(Xsamp)
    nlab = len(set(db)) - (1 if -1 in db else 0)
    nr = int((db == -1).sum())
    if nlab >= 2:
        m = db != -1
        sil = silhouette_score(Xsamp[m], db[m])
        print(f"  DBSCAN eps=0.6 min_samples=10  clusters={nlab}  ruido={nr}  "
              f"silhouette(sin ruido)={sil:.4f}")
    else:
        print(f"  DBSCAN eps=0.6 min_samples=10  clusters={nlab}  ruido={nr}  (silhouette N/A)")

    # -- (c) RFM sobre la muestra de Online Retail II + clustering --
    print("\n== (c) RFM sobre muestra de Online Retail II + Ward ==")
    df = obtener_retail_muestra()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Amount"] = df["Quantity"] * df["Price"]
    ref = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda x: (ref - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("Amount", "sum")).reset_index()
    print(f"  RFM: {rfm.shape[0]} clientes  "
          f"R[{rfm.Recency.min()}-{rfm.Recency.max()}]  "
          f"F[{rfm.Frequency.min()}-{rfm.Frequency.max()}]  "
          f"M[{rfm.Monetary.min():.0f}-{rfm.Monetary.max():.0f}]")
    Xr = StandardScaler().fit_transform(
        np.log1p(rfm[["Recency", "Frequency", "Monetary"]].values))
    for k in (3, 4, 5):
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xr)
        print(f"  Ward k={k}  silhouette={silhouette_score(Xr, lab):.4f}  "
              f"tam={np.bincount(lab)}")

    print("\n  (Los targets con tolerancia los fija investigador-tema en "
          "la ficha de réplica del paper; aqui solo se OBSERVAN.)")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    print(f"Directorio de datos (OneDrive): {DATA_DIR}")
    print(f"Cache del completo (temporal):  {CACHE_DIR}\n")

    mam = obtener_mammography()
    print()
    sh = obtener_shuttle()
    print()
    muestra = obtener_retail_muestra()

    print("\n== Resumen ==")
    print(f"mammography.csv               : {mam.shape[0]} x {mam.shape[1]}  "
          f"(anom {mam['anomalia'].mean()*100:.2f}%)")
    print(f"shuttle_odds.csv              : {sh.shape[0]} x {sh.shape[1]}  "
          f"(anom {sh['anomalia'].mean()*100:.2f}%)")
    print(f"online_retail_II_muestra.csv  : {muestra.shape[0]} x {muestra.shape[1]}  "
          f"({muestra['Customer ID'].nunique()} clientes; COMPLETO no en OneDrive)")

    print("\n== Checksums SHA256 ==")
    for p in (MAMMO_FILE, SHUTTLE_FILE, RETAIL_SAMPLE):
        if os.path.exists(p):
            print(f"  {os.path.basename(p):30s} {sha256_file(p)}")

    if "--full-retail" in argv:
        cargar_retail_completo_limpio()
    if "--smoke" in argv:
        smoke()
    print("\nVerificacion de fuentes: 18/07/2026. Todo OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
