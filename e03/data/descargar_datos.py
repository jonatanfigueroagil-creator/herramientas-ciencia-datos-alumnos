# -*- coding: utf-8 -*-
"""
Descarga y verificación de los datasets de la Sesión EPE E3
(Demasiadas variables: regularización y PCA) del curso UPC
"Herramientas para la Ciencia de Datos".

Dos casos de negocio (los que fija el sílabo EPE, Sesión 3):

  1) COMMUNITIES AND CRIME  (regularización / "demasiadas variables").
     1994 comunidades de EE. UU. descritas por ~100 predictores
     socioeconómicos y policiales (todos normalizados a [0,1]); la variable
     objetivo es 'ViolentCrimesPerPop' (crímenes violentos por 100 000 hab.).
     Es el caso de ALTA DIMENSIÓN: muchos predictores correlacionados -> ideal
     para comparar OLS vs. Ridge/LASSO y ver cómo el LASSO selecciona
     variables. Fuente del sílabo: UCI id 183 (Redmond, 2009).
     Si existe una copia local ya verificada, se reutiliza (mismo checksum) en
     vez de volver a descargar; si no está, se descarga del mirror abierto de UCI.

  2) WINE  (PCA / comprimir y visualizar).  178 vinos descritos por 13 medidas
     químicas (alcohol, fenoles, color, etc.) de tres cultivares. Fuente del
     sílabo: UCI id 109; integrado en scikit-learn (load_wine). Es el caso para
     resumir muchas variables en pocas componentes y visualizar en 2D.
     Si existe una copia local ya verificada, se reutiliza en vez de volver a
     generarla; si no está, se genera con sklearn.datasets.load_wine (mismo
     esquema, 178 x 14).

Ambos archivos pesan < 1.2 MB -> se conservan en OneDrive (la regla de
exclusión aplica solo a >25 MB; no hace falta muestra).

Idempotente: si el CSV ya existe y su esquema (filas x columnas + columnas
clave) es correcto, NO vuelve a obtenerse. Funciona en local (Windows/venv) y
en Google Colab (usa el directorio de trabajo actual).

Uso:
    "<python_del_venv>" descargar_datos.py
    # o en Colab:  %run descargar_datos.py

Fecha de verificación de fuentes: 06/08/2026.
"""
import hashlib
import io
import os
import shutil
import sys

try:
    import requests
except ImportError:  # en Colab requests viene preinstalado
    requests = None

import pandas as pd

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (curso UPC E03 descargar_datos)"}

if "google.colab" in sys.modules:
    DATA_DIR = os.getcwd()
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

COMM_FILE = os.path.join(DATA_DIR, "communities.csv")   # 1994 x 128 (regularización)
WINE_FILE = os.path.join(DATA_DIR, "wine.csv")          # 178 x 14   (PCA)

# Copia local ya verificada (si existe), para evitar re-descargar/regenerar.
_CACHE_COMM_DIR = os.path.normpath(os.path.join(
    DATA_DIR, "..", "..", "..", "Sesiones", "S05_regularizacion", "data"))
_CACHE_WINE_DIR = os.path.normpath(os.path.join(
    DATA_DIR, "..", "..", "..", "Sesiones", "S06_reduccion_dim", "data"))
CACHE_COMM_CSV = os.path.join(_CACHE_COMM_DIR, "communities.csv")
CACHE_WINE_CSV = os.path.join(_CACHE_WINE_DIR, "wine.csv")

# Fuente abierta de communities: UCI id 183.
COMM_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "communities/communities.data"
)

# communities.data NO trae cabecera; el orden de columnas es el de los
# 128 @attribute de communities.names.
COMM_COLS = [
    "state", "county", "community", "communityname", "fold",
    "population", "householdsize", "racepctblack", "racePctWhite",
    "racePctAsian", "racePctHisp", "agePct12t21", "agePct12t29",
    "agePct16t24", "agePct65up", "numbUrban", "pctUrban", "medIncome",
    "pctWWage", "pctWFarmSelf", "pctWInvInc", "pctWSocSec", "pctWPubAsst",
    "pctWRetire", "medFamInc", "perCapInc", "whitePerCap", "blackPerCap",
    "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap",
    "NumUnderPov", "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad",
    "PctBSorMore", "PctUnemployed", "PctEmploy", "PctEmplManu",
    "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf", "MalePctDivorce",
    "MalePctNevMarr", "FemalePctDiv", "TotalPctDiv", "PersPerFam",
    "PctFam2Par", "PctKids2Par", "PctYoungKids2Par", "PctTeen2Par",
    "PctWorkMomYoungKids", "PctWorkMom", "NumIlleg", "PctIlleg", "NumImmig",
    "PctImmigRecent", "PctImmigRec5", "PctImmigRec8", "PctImmigRec10",
    "PctRecentImmig", "PctRecImmig5", "PctRecImmig8", "PctRecImmig10",
    "PctSpeakEnglOnly", "PctNotSpeakEnglWell", "PctLargHouseFam",
    "PctLargHouseOccup", "PersPerOccupHous", "PersPerOwnOccHous",
    "PersPerRentOccHous", "PctPersOwnOccup", "PctPersDenseHous",
    "PctHousLess3BR", "MedNumBR", "HousVacant", "PctHousOccup",
    "PctHousOwnOcc", "PctVacantBoarded", "PctVacMore6Mos", "MedYrHousBuilt",
    "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart", "OwnOccMedVal",
    "OwnOccHiQuart", "RentLowQ", "RentMedian", "RentHighQ", "MedRent",
    "MedRentPctHousInc", "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg",
    "NumInShelters", "NumStreet", "PctForeignBorn", "PctBornSameState",
    "PctSameHouse85", "PctSameCity85", "PctSameState85", "LemasSwornFT",
    "LemasSwFTPerPop", "LemasSwFTFieldOps", "LemasSwFTFieldPerPop",
    "LemasTotalReq", "LemasTotReqPerPop", "PolicReqPerOffic", "PolicPerPop",
    "RacialMatchCommPol", "PctPolicWhite", "PctPolicBlack", "PctPolicHisp",
    "PctPolicAsian", "PctPolicMinor", "OfficAssgnDrugUnits",
    "NumKindsDrugsSeiz", "PolicAveOTWorked", "LandArea", "PopDens",
    "PctUsePubTrans", "PolicCars", "PolicOperBudg", "LemasPctPolicOnPatr",
    "LemasGangUnitDeploy", "LemasPctOfficDrugUn", "PolicBudgPerPop",
    "ViolentCrimesPerPop",
]

# Esquemas esperados
COMM_KEY_COLS = ["state", "communityname", "fold", "ViolentCrimesPerPop"]
COMM_ROWS, COMM_COLS_N = 1994, 128
WINE_KEY_COLS = ["alcohol", "flavanoids", "color_intensity", "proline", "target"]
WINE_ROWS, WINE_COLS_N = 178, 14


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str) -> bytes:
    if requests is None:
        raise RuntimeError("requests no disponible")
    r = requests.get(url, headers=HEADERS, timeout=90)
    r.raise_for_status()
    return r.content


def _verificar_esquema(df, key_cols, nrows, ncols, nombre):
    faltan = [c for c in key_cols if c not in df.columns]
    if faltan:
        raise ValueError(f"[{nombre}] faltan columnas clave {faltan}; "
                         f"hay {df.shape[1]} columnas")
    if nrows is not None and len(df) != nrows:
        raise ValueError(f"[{nombre}] se esperaban {nrows} filas y hay {len(df)}")
    if ncols is not None and df.shape[1] != ncols:
        raise ValueError(f"[{nombre}] se esperaban {ncols} columnas y hay {df.shape[1]}")


def _csv_valido(path, key_cols, nrows, ncols) -> bool:
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path, low_memory=False)
        _verificar_esquema(df, key_cols, nrows, ncols, os.path.basename(path))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ! {os.path.basename(path)} existe pero no valida ({e}); se re-obtiene.")
        return False


# ---------------------------------------------------------------------------
# COMMUNITIES AND CRIME (regularización / alta dimensión)
# ---------------------------------------------------------------------------
def obtener_communities() -> pd.DataFrame:
    print("== COMMUNITIES AND CRIME 1994x128 (regularización; UCI 183) ==")
    if _csv_valido(COMM_FILE, COMM_KEY_COLS, COMM_ROWS, COMM_COLS_N):
        print(f"  = presente y verificado: {COMM_FILE}")
        return pd.read_csv(COMM_FILE, low_memory=False)

    # 1) copia local ya verificada (evita re-descargar)
    if _csv_valido(CACHE_COMM_CSV, COMM_KEY_COLS, COMM_ROWS, COMM_COLS_N):
        shutil.copyfile(CACHE_COMM_CSV, COMM_FILE)
        print(f"  + copiado desde cache local -> {COMM_FILE}")
        return pd.read_csv(COMM_FILE, low_memory=False)

    # 2) descarga del mirror abierto de UCI (faltantes '?' -> NaN)
    raw = _fetch(COMM_DATA_URL)
    df = pd.read_csv(io.BytesIO(raw), header=None, names=COMM_COLS,
                     na_values="?", low_memory=False)
    _verificar_esquema(df, COMM_KEY_COLS, COMM_ROWS, COMM_COLS_N, "UCI communities")
    df.to_csv(COMM_FILE, index=False, encoding="utf-8")
    print(f"  + descargado de UCI -> {COMM_FILE}")
    print(f"    {COMM_DATA_URL}")
    return df


# ---------------------------------------------------------------------------
# WINE (PCA) -- copia local o load_wine
# ---------------------------------------------------------------------------
def obtener_wine() -> pd.DataFrame:
    print("== WINE 178x14 (PCA; UCI 109 / load_wine) ==")
    if _csv_valido(WINE_FILE, WINE_KEY_COLS, WINE_ROWS, WINE_COLS_N):
        print(f"  = presente y verificado: {WINE_FILE}")
        return pd.read_csv(WINE_FILE)

    # 1) copia local ya verificada
    if _csv_valido(CACHE_WINE_CSV, WINE_KEY_COLS, WINE_ROWS, WINE_COLS_N):
        shutil.copyfile(CACHE_WINE_CSV, WINE_FILE)
        print(f"  + copiado desde cache local -> {WINE_FILE}")
        return pd.read_csv(WINE_FILE)

    # 2) generar con sklearn.datasets.load_wine (mismo conjunto UCI id 109)
    from sklearn.datasets import load_wine
    d = load_wine(as_frame=True)
    df = d.frame  # 13 features + target
    _verificar_esquema(df, WINE_KEY_COLS, WINE_ROWS, WINE_COLS_N, "load_wine")
    df.to_csv(WINE_FILE, index=False, encoding="utf-8")
    print(f"  + generado via load_wine -> {WINE_FILE}")
    return df


# ---------------------------------------------------------------------------
def main() -> int:
    print(f"Directorio de datos: {DATA_DIR}\n")
    comm = obtener_communities()
    print()
    wine = obtener_wine()
    print("\n== Resumen ==")
    print(f"communities.csv (regularización) : {comm.shape[0]} x {comm.shape[1]}")
    print(f"wine.csv        (PCA)            : {wine.shape[0]} x {wine.shape[1]}")
    print("\n== Checksums SHA256 ==")
    for p in (COMM_FILE, WINE_FILE):
        if os.path.exists(p):
            print(f"  {os.path.basename(p):18s} {sha256_file(p)}")
    print("\nVerificación de fuentes: 06/08/2026. Todo OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
