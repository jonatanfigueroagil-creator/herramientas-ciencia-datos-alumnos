# -*- coding: utf-8 -*-
"""
Descarga y verificacion de los datos de la Sesion S08
(Reglas de asociacion + market basket analysis) del curso UPC
"Herramientas para la Ciencia de Datos".

DOS conjuntos:

  1) T10I4D100K (REPLICACION / benchmark FIMI) -> "T10I4D100K.dat"  100000 trans.
     Dataset transaccional SINTETICO generado por el IBM Quest Synthetic Data
     Generator (Agrawal & Srikant 1994), benchmark canonico del repositorio
     FIMI (Frequent Itemset Mining Implementations, U. Antwerpen).
       - T10  = longitud media de transaccion 10 items.
       - I4   = longitud media de los itemsets potencialmente frecuentes 4.
       - D100K = 100 000 transacciones.
     Formato: UNA linea por transaccion (ticket); items = enteros separados por
     espacio (0..999; solo 870 aparecen). Es una lista de transacciones CRUDA,
     NO una matriz: no tiene cabecera ni comas.
     Verificado 18/07/2026: 100 000 transacciones, 870 items distintos,
     longitud media 10.10 (min 1, max 29), 1 010 228 items en total.
     Fuente del silabo: http://fimi.uantwerpen.be/data/  -> HTTP-404 (18/07/2026).
     FALLBACK aplicado: mirror abierto en GitHub del mismo archivo (contenido
     identico al T10I4D100K.dat de FIMI, aunque el repo lo nombra .csv):
       narendrababu-um/t10i4d100kdatapac, data-raw/T10I4D100K.csv
     Cita: Agrawal, R. & Srikant, R. (1994). "Fast Algorithms for Mining
     Association Rules". Proc. 20th VLDB. Generador IBM Quest.
     Licencia: benchmark academico de dominio publico (datos sinteticos).
     Pesa ~3.8 MB (<25 MB) -> se CONSERVA en la carpeta.

  2) ONLINE RETAIL II — CANASTAS (negocio / market basket) ->
     "online_retail_baskets_muestra.csv"
     Se REUTILIZA la fuente ya verificada en S07 (UCI id 502). El COMPLETO son
     ~1.07M filas en un .xlsx de ~45.6 MB -> NUNCA se guarda en OneDrive: se baja
     el .zip de UCI SOLO en runtime (a un temporal del sistema), se limpia y se
     guarda una MUESTRA <=1 MB.
     A diferencia de la muestra RFM de S07 (que agrupa por Customer ID y NO trae
     la identidad del producto), para market basket la unidad de analisis es la
     FACTURA y hace falta la descripcion del producto. Por eso esta muestra se
     construye por FACTURA (Invoice) y conserva la columna Description:
       transaccion = lista de Description de una misma Invoice.
     Muestra: N facturas elegidas al azar (RandomState=42) con TODAS sus lineas;
     columnas [Invoice, StockCode, Description, Country]. Cada factura es una
     canasta completa (todos sus productos), apta para one-hot por ticket.
       - Fuente: UCI id 502, https://archive.ics.uci.edu/dataset/502/online+retail+ii
         zip: https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
         SHA256(zip) = 572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb
       - Limpieza: quitar faltantes de Customer ID, cancelaciones (Invoice que
         empieza por 'C'), Quantity<=0, Price<=0 y Description vacia.
     Cita: Chen, D. (2019). Online Retail II. UCI ML Repository.
     https://doi.org/10.24432/C5CG6D . Licencia CC BY 4.0.

Idempotente: si un archivo ya existe y su esquema es correcto, NO se regenera.
Funciona en local (Windows/venv) y en Google Colab. El .xlsx completo de Online
Retail II se descarga a un temporal FUERA de OneDrive y NO se conserva.

Uso:
    "<python_del_venv>" descargar_datos.py            # genera + verifica esquema
    "<python_del_venv>" descargar_datos.py --smoke     # ademas corre el smoke mlxtend
    "<python_del_venv>" descargar_datos.py --full-retail  # baja+limpia el completo
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
HEADERS = {"User-Agent": "Mozilla/5.0 (curso UPC S08 descargar_datos)"}

if "google.colab" in sys.modules:
    DATA_DIR = os.getcwd()
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

CACHE_DIR = os.path.join(tempfile.gettempdir(), "curso_upc_s08_datos")

# --- 1) T10I4D100K (benchmark FIMI) ---
T10_FILE = os.path.join(DATA_DIR, "T10I4D100K.dat")
T10_URL = ("https://raw.githubusercontent.com/narendrababu-um/"
           "t10i4d100kdatapac/master/data-raw/T10I4D100K.csv")
T10_SHA256 = "464d7b87aa7fdc0dc432ddad0493515569b2968b0f229ce39399c454a8c62e0c"
T10_N_TRANS = 100000        # transacciones esperadas
T10_N_ITEMS = 870           # items distintos esperados

# --- 2) Online Retail II — canastas (negocio) ---
RETAIL_BASKETS = os.path.join(DATA_DIR, "online_retail_baskets_muestra.csv")
RETAIL_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
)
RETAIL_ZIP_SHA256 = (
    "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"
)
RETAIL_N_FACTURAS = 800         # facturas muestreadas -> muestra <=1 MB
#   (~21.5 lineas/factura, ~56 bytes/fila => ~0.96 MB; el limite OneDrive es 1 MB)
RETAIL_RANDOM_STATE = 42
RETAIL_COLS = ["Invoice", "StockCode", "Description", "Country"]


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


# ---------------------------------------------------------------------------
# 1) T10I4D100K
# ---------------------------------------------------------------------------
def _t10_valido() -> bool:
    """Idempotencia: archivo presente y con nº de lineas correcto."""
    if not os.path.exists(T10_FILE):
        return False
    n = sum(1 for _ in open(T10_FILE, encoding="utf-8"))
    if n != T10_N_TRANS:
        print(f"  ! T10I4D100K.dat: {n} lineas != {T10_N_TRANS}; se regenera.")
        return False
    return True


def obtener_t10i4d100k() -> list:
    """Descarga (si hace falta) T10I4D100K.dat y devuelve la lista de transacciones."""
    print("== T10I4D100K 100000 trans. (benchmark FIMI; Agrawal & Srikant 1994) ==")
    if not _t10_valido():
        print(f"  ... descargando mirror GitHub (FIMI original da 404)\n      {T10_URL}")
        raw = _fetch(T10_URL)
        sha = sha256_bytes(raw)
        aviso = "" if sha == T10_SHA256 else "  [SHA256 de fuente NUEVO -> revisar]"
        # normalizar saltos de linea a \n y escribir tal cual (formato .dat crudo)
        txt = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        with open(T10_FILE, "w", encoding="utf-8", newline="\n") as f:
            f.write(txt)
        print(f"  + guardado -> {T10_FILE}  ({len(raw)/1e6:.2f} MB)"
              f"  SHA256(fuente)={sha[:16]}...{aviso}")

    trans = parse_transacciones(T10_FILE)
    n_items = len({it for t in trans for it in t})
    long_media = sum(len(t) for t in trans) / len(trans)
    if len(trans) != T10_N_TRANS:
        raise ValueError(f"nº transacciones {len(trans)} != {T10_N_TRANS}")
    if n_items != T10_N_ITEMS:
        raise ValueError(f"nº items distintos {n_items} != {T10_N_ITEMS}")
    print(f"  = verificado: {len(trans)} transacciones, {n_items} items distintos, "
          f"long. media {long_media:.2f}")
    return trans


def parse_transacciones(path: str) -> list:
    """Parsea un .dat transaccional (una linea por ticket, items enteros
    separados por espacio) a una lista de listas de enteros."""
    trans = []
    with open(path, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                trans.append([int(x) for x in linea.split()])
    return trans


# ---------------------------------------------------------------------------
# 2) ONLINE RETAIL II — canastas por factura
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
    open(zp, "wb").write(raw)
    print(f"  ... zip -> {zp}  SHA256={sha256_bytes(raw)[:16]}...")
    return raw


def cargar_retail_completo_limpio() -> pd.DataFrame:
    """DataFrame COMPLETO limpio, con Description (para canastas). No se guarda
    en OneDrive; se cachea limpio en un temporal para acelerar re-ejecuciones."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    pk = os.path.join(CACHE_DIR, "retail_limpio_baskets.pkl")
    if os.path.exists(pk):
        print(f"  = retail limpio en cache temporal ({pk})")
        return pd.read_pickle(pk)
    raw = _descargar_retail_zip()
    z = zipfile.ZipFile(io.BytesIO(raw))
    name = [n for n in z.namelist() if n.lower().endswith(".xlsx")][0]
    print(f"  ... leyendo {name} (dos hojas, ~1.07M filas; tarda ~1-2 min)")
    sheets = pd.read_excel(io.BytesIO(z.read(name)), sheet_name=None, engine="openpyxl")
    df = pd.concat(sheets.values(), ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    n0 = len(df)
    df = df[df["Customer ID"].notna()]
    df = df[~df["Invoice"].astype(str).str.startswith("C")]      # cancelaciones
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]            # devoluciones/precio 0
    df = df[df["Description"].notna()]
    df["Description"] = df["Description"].astype(str).str.strip()
    df = df[df["Description"] != ""]
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    print(f"  ... limpieza: {n0} -> {len(df)} filas  "
          f"({df['Invoice'].nunique()} facturas, {df['Description'].nunique()} productos)")
    df = df.reset_index(drop=True)
    df.to_pickle(pk)
    return df


def obtener_retail_baskets() -> pd.DataFrame:
    print("== ONLINE RETAIL II — CANASTAS (negocio; UCI 502; completo NUNCA en OneDrive) ==")
    if os.path.exists(RETAIL_BASKETS):
        df = pd.read_csv(RETAIL_BASKETS)
        if (set(RETAIL_COLS).issubset(df.columns)
                and df["Invoice"].nunique() == RETAIL_N_FACTURAS):
            print(f"  = presente y verificado: {RETAIL_BASKETS} "
                  f"({len(df)} filas, {df['Invoice'].nunique()} facturas)")
            return df
        print("  ! esquema/nº de facturas distinto; se regenera la muestra.")

    full = cargar_retail_completo_limpio()
    facturas = np.sort(full["Invoice"].unique())
    rng = np.random.RandomState(RETAIL_RANDOM_STATE)
    sel = rng.choice(facturas, size=RETAIL_N_FACTURAS, replace=False)
    muestra = (full[full["Invoice"].isin(sel)][RETAIL_COLS]
               .sort_values(["Invoice", "StockCode"]).reset_index(drop=True))
    muestra.to_csv(RETAIL_BASKETS, index=False, encoding="utf-8")
    mb = os.path.getsize(RETAIL_BASKETS) / 1e6
    print(f"  + muestra {RETAIL_N_FACTURAS} facturas (rs={RETAIL_RANDOM_STATE}) -> "
          f"{RETAIL_BASKETS} ({len(muestra)} filas, {mb:.3f} MB)")
    if mb > 1.0:
        print(f"  ! AVISO: la muestra pesa {mb:.3f} MB (>1 MB). Reducir "
              f"RETAIL_N_FACTURAS.")
    return muestra


def canastas_por_factura(df: pd.DataFrame) -> list:
    """Convierte el DataFrame largo (una fila por linea de factura) en una lista
    de transacciones = por cada Invoice, la lista de sus Description."""
    return (df.groupby("Invoice")["Description"]
              .apply(lambda s: sorted(set(s)))
              .tolist())


# ---------------------------------------------------------------------------
# SMOKE de VIABILIDAD (reporta valores OBSERVADOS; NO define targets)
# ---------------------------------------------------------------------------
def smoke():
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

    print("\n########## SMOKE DE VIABILIDAD S08 (mlxtend) ##########")

    # -- (a) T10I4D100K: Apriori vs FP-Growth deben COINCIDIR --
    print("\n== (a) T10I4D100K: one-hot bool + Apriori vs FP-Growth (min_support=0.01) ==")
    trans = obtener_t10i4d100k()
    te = TransactionEncoder()
    ary = te.fit_transform(trans)                     # bool por defecto
    onehot = pd.DataFrame(ary, columns=te.columns_)   # dtype bool
    print(f"  one-hot: {onehot.shape[0]} x {onehot.shape[1]}  dtype={onehot.dtypes.iloc[0]}")

    import time
    t0 = time.perf_counter()
    fi_ap = apriori(onehot, min_support=0.01, use_colnames=True)
    t_ap = time.perf_counter() - t0
    t0 = time.perf_counter()
    fi_fp = fpgrowth(onehot, min_support=0.01, use_colnames=True)
    t_fp = time.perf_counter() - t0
    print(f"  Apriori   : {len(fi_ap)} itemsets frecuentes  ({t_ap:.2f}s)")
    print(f"  FP-Growth : {len(fi_fp)} itemsets frecuentes  ({t_fp:.2f}s)")
    coincide = len(fi_ap) == len(fi_fp)
    # comparar los conjuntos exactos, no solo el conteo
    set_ap = set(frozenset(x) for x in fi_ap["itemsets"])
    set_fp = set(frozenset(x) for x in fi_fp["itemsets"])
    print(f"  COINCIDEN (conteo)={coincide}  (conjuntos identicos)={set_ap == set_fp}")
    by_len = fi_ap["itemsets"].apply(len).value_counts().sort_index()
    print("  itemsets por tamaño:", {int(k): int(v) for k, v in by_len.items()})

    reglas = association_rules(fi_ap, metric="confidence", min_threshold=0.5)
    print(f"  Reglas (confidence>=0.5): {len(reglas)}")
    if len(reglas):
        top = reglas.sort_values("lift", ascending=False).head(3)
        for _, r in top.iterrows():
            a = ",".join(map(str, r["antecedents"]))
            c = ",".join(map(str, r["consequents"]))
            print(f"    {a} -> {c}  sup={r['support']:.4f} conf={r['confidence']:.3f} "
                  f"lift={r['lift']:.2f}")

    # -- (b) Retail: canasta por factura + apriori pequeño --
    print("\n== (b) ONLINE RETAIL II: canasta por factura + Apriori (min_support=0.02) ==")
    dfr = obtener_retail_baskets()
    canastas = canastas_por_factura(dfr)
    print(f"  {len(canastas)} canastas; tamaño medio "
          f"{np.mean([len(c) for c in canastas]):.1f} productos")
    te2 = TransactionEncoder()
    oh2 = pd.DataFrame(te2.fit_transform(canastas), columns=te2.columns_)
    print(f"  one-hot retail: {oh2.shape[0]} x {oh2.shape[1]} productos distintos")
    fi_r = apriori(oh2, min_support=0.02, use_colnames=True)
    fi_r_fp = fpgrowth(oh2, min_support=0.02, use_colnames=True)
    print(f"  Apriori: {len(fi_r)} itemsets  |  FP-Growth: {len(fi_r_fp)} itemsets  "
          f"(coinciden={len(fi_r) == len(fi_r_fp)})")
    if len(fi_r):
        reglas_r = association_rules(fi_r, metric="confidence", min_threshold=0.3)
        print(f"  Reglas retail (confidence>=0.3): {len(reglas_r)}")
        if len(reglas_r):
            top = reglas_r.sort_values("lift", ascending=False).head(3)
            for _, r in top.iterrows():
                a = " + ".join(map(str, r["antecedents"]))
                c = " + ".join(map(str, r["consequents"]))
                print(f"    [{a}] -> [{c}]  conf={r['confidence']:.2f} lift={r['lift']:.2f}")

    print("\n  (Los targets con tolerancia los fija investigador-tema en "
          "la ficha de réplica del paper; aqui solo se OBSERVAN.)")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    print(f"Directorio de datos (OneDrive): {DATA_DIR}")
    print(f"Cache del completo (temporal):  {CACHE_DIR}\n")

    trans = obtener_t10i4d100k()
    print()
    dfr = obtener_retail_baskets()

    print("\n== Resumen ==")
    print(f"T10I4D100K.dat                     : {len(trans)} transacciones "
          f"(benchmark FIMI, se conserva)")
    print(f"online_retail_baskets_muestra.csv  : {len(dfr)} filas, "
          f"{dfr['Invoice'].nunique()} facturas (COMPLETO no en OneDrive)")

    print("\n== Checksums SHA256 ==")
    for p in (T10_FILE, RETAIL_BASKETS):
        if os.path.exists(p):
            print(f"  {os.path.basename(p):34s} {sha256_file(p)}")

    if "--full-retail" in argv:
        cargar_retail_completo_limpio()
    if "--smoke" in argv:
        smoke()
    print("\nVerificacion de fuentes: 18/07/2026. Todo OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
