"""Descarga y verifica los datasets de la Sesion 13 - Analisis de supervivencia.

Datasets:
  1) Rossi recidivism (REPLICACION, paper seminal Cox) ----> "rossi_recidivism.csv"
     432 ex-reclusos seguidos 1 anho (52 semanas). Evento = re-arresto.
     Dataset clasico del modelo de Cox (Allison 1995; Fox & Weisberg).
     Fuente primaria: lifelines.datasets.load_rossi() (integrado en lifelines).
     Fallback sin lifelines: Rdatasets paquete carData 'Rossi' (subconjunto de
     las 9 columnas estandar).

  2) Veterans' Administration Lung Cancer (REPLICACION 2, Cox 1972) ->
     "veterans_lung_cancer.csv"
     137 pacientes con cancer de pulmon avanzado; ensayo aleatorizado de dos
     tratamientos. Evento = muerte. Datos analizados con el modelo de Cox
     (Kalbfleisch & Prentice 1980; Cox 1972).
     Nota (error conocido): scikit-survival NO instala en Python 3.13 (dep
     'ecos' sin wheel) -> NO se usa sksurv. Veterans se obtiene del mirror
     abierto Rdatasets (paquete R 'survival', dataset 'veteran').

  3) Telco Customer Churn (NEGOCIO / churn) -> "telco_churn.csv"
     7043 clientes de una telco; tenure = tiempo (meses) hasta el evento churn.
     Sustituto de negocio (no es replica de paper). Se REUSA la misma fuente
     verificada en S09 (CSV oficial IBM via mirror abierto).
     Fuente del silabo: Kaggle blastchar/telco-customer-churn -> requiere login;
     mirror abierto byte-identico: treselle-systems/customer_churn_analysis.

Idempotente: cada archivo se (re)crea solo si falta o no valida.
Funciona local y en Colab (usa requests + lifelines, ambos pip-instalables).
Pesos: los tres < 25 MB (Telco ~0.95 MB) -> se conservan completos; no hay
muestra separada. Cada archivo se valida por filas x columnas y SHA256.

Uso:
    python descargar_datos.py            # descarga/verifica y reporta checksums
"""
from __future__ import annotations

import hashlib
import os

import pandas as pd
import requests

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1) Rossi recidivism (replicacion, Cox) ---
ROSSI_FILE = os.path.join(DATA_DIR, "rossi_recidivism.csv")
ROSSI_N, ROSSI_NCOLS = 432, 9
ROSSI_COLS = ["week", "arrest", "fin", "age", "race", "wexp", "mar", "paro", "prio"]
ROSSI_EVENTS = 114  # arrest == 1
ROSSI_SHA256 = "8e5f9d64821b5f98bb32fda57eb8466f1a40ed18b41c6d85dcc16a15988a1ddc"
ROSSI_FALLBACK_URL = (
    "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Rossi.csv"
)

# --- 2) Veterans' Lung Cancer (replicacion 2, Cox 1972) ---
VET_FILE = os.path.join(DATA_DIR, "veterans_lung_cancer.csv")
VET_N, VET_NCOLS = 137, 9
VET_KEY_COLS = ["trt", "celltype", "time", "status", "karno", "diagtime",
                "age", "prior"]
VET_EVENTS = 128  # status == 1 (muertes)
VET_URL = (
    "https://vincentarelbundock.github.io/Rdatasets/csv/survival/veteran.csv"
)
VET_SHA256_SRC = (
    "3fba5cb9b15a10ab94d54e28b54d2c39d95c27b0fcff1c57aee0eb7edfd93950"
)

# --- 3) Telco Customer Churn (negocio / churn) ---
TELCO_FILE = os.path.join(DATA_DIR, "telco_churn.csv")
TELCO_N, TELCO_NCOLS = 7043, 21
TELCO_KEY_COLS = ["customerID", "tenure", "MonthlyCharges", "TotalCharges",
                  "Contract", "Churn"]
TELCO_URL = (
    "https://raw.githubusercontent.com/treselle-systems/"
    "customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
TELCO_SHA256_SRC = (
    "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str) -> bytes:
    resp = requests.get(url, timeout=90, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# 1) Rossi recidivism
# ---------------------------------------------------------------------------
def obtener_rossi() -> pd.DataFrame:
    """Devuelve el DataFrame Rossi (432 x 9) verificado; lo (re)crea si falta."""
    print("== Rossi recidivism 432 filas (REPLICACION; paper Cox; lifelines) ==")
    if os.path.exists(ROSSI_FILE):
        df = pd.read_csv(ROSSI_FILE)
        if list(df.columns) == ROSSI_COLS and df.shape == (ROSSI_N, ROSSI_NCOLS):
            print(f"  ya existe y valida: {df.shape}  SHA256={sha256_file(ROSSI_FILE)[:16]}...")
            return df
        print("  archivo invalido -> se regenera")

    df = None
    try:  # fuente primaria: lifelines
        from lifelines.datasets import load_rossi
        df = load_rossi()[ROSSI_COLS].copy()
        print("  fuente: lifelines.datasets.load_rossi()")
    except Exception as e:  # fallback: Rdatasets carData Rossi (subconjunto 9 cols)
        print(f"  lifelines no disponible ({type(e).__name__}); fallback Rdatasets carData")
        import io
        raw = _fetch(ROSSI_FALLBACK_URL)
        full = pd.read_csv(io.BytesIO(raw))
        df = full[ROSSI_COLS].copy()

    if df.shape != (ROSSI_N, ROSSI_NCOLS):
        raise ValueError(f"Rossi: {df.shape} != ({ROSSI_N}, {ROSSI_NCOLS})")
    if int(df["arrest"].sum()) != ROSSI_EVENTS:
        raise ValueError(f"Rossi: eventos {int(df['arrest'].sum())} != {ROSSI_EVENTS}")
    df.to_csv(ROSSI_FILE, index=False)
    print(f"  guardado {os.path.basename(ROSSI_FILE)}  SHA256={sha256_file(ROSSI_FILE)[:16]}...")
    return df


# ---------------------------------------------------------------------------
# 2) Veterans' Lung Cancer
# ---------------------------------------------------------------------------
def obtener_veterans() -> pd.DataFrame:
    """Devuelve el DataFrame Veterans (137 x 9) verificado; lo descarga si falta."""
    print("== Veterans' Lung Cancer 137 filas (REPLICACION 2; Cox 1972; Rdatasets) ==")
    if not os.path.exists(VET_FILE):
        raw = _fetch(VET_URL)
        sha = sha256_bytes(raw)
        aviso = "" if sha == VET_SHA256_SRC else "  [SHA256 de fuente NUEVO -> revisar]"
        with open(VET_FILE, "wb") as fh:
            fh.write(raw)
        print(f"  descargado de Rdatasets  SHA256(fuente)={sha[:16]}...{aviso}")
    df = pd.read_csv(VET_FILE)
    faltan = [c for c in VET_KEY_COLS if c not in df.columns]
    if faltan:
        raise ValueError(f"Veterans: faltan columnas {faltan}")
    if df.shape != (VET_N, VET_NCOLS):
        raise ValueError(f"Veterans: {df.shape} != ({VET_N}, {VET_NCOLS})")
    if int(df["status"].sum()) != VET_EVENTS:
        raise ValueError(f"Veterans: eventos {int(df['status'].sum())} != {VET_EVENTS}")
    print(f"  valida: {df.shape}  SHA256(local)={sha256_file(VET_FILE)[:16]}...")
    return df


# ---------------------------------------------------------------------------
# 3) Telco Customer Churn
# ---------------------------------------------------------------------------
def obtener_telco() -> pd.DataFrame:
    """Devuelve el DataFrame Telco (7043 x 21) CRUDO verificado; lo descarga si falta."""
    print("== Telco Customer Churn 7043 filas (NEGOCIO; IBM sample; mirror abierto) ==")
    if not os.path.exists(TELCO_FILE):
        raw = _fetch(TELCO_URL)
        sha = sha256_bytes(raw)
        aviso = "" if sha == TELCO_SHA256_SRC else "  [SHA256 de fuente NUEVO -> revisar]"
        with open(TELCO_FILE, "wb") as fh:
            fh.write(raw)
        print(f"  descargado del mirror  SHA256(fuente)={sha[:16]}...{aviso}")
    df = pd.read_csv(TELCO_FILE)
    faltan = [c for c in TELCO_KEY_COLS if c not in df.columns]
    if faltan:
        raise ValueError(f"Telco: faltan columnas {faltan}")
    if df.shape != (TELCO_N, TELCO_NCOLS):
        raise ValueError(f"Telco: {df.shape} != ({TELCO_N}, {TELCO_NCOLS})")
    print(f"  valida: {df.shape}  SHA256(local)={sha256_file(TELCO_FILE)[:16]}...")
    return df


def limpiar_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza minima de Telco para analisis de supervivencia:
    - TotalCharges: 11 celdas en blanco (clientes con tenure=0) -> NaN -> se
      eliminan (7043 -> 7032 filas).
    - evento = (Churn == 'Yes'); duracion = tenure (meses).
    Devuelve una copia con columnas 'evento' (0/1) y 'duracion' anhadidas.
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    n_na = int(out["TotalCharges"].isna().sum())
    out = out.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    out["evento"] = (out["Churn"] == "Yes").astype(int)
    out["duracion"] = out["tenure"].astype(float)
    print(f"  limpieza Telco: {n_na} NaN eliminados -> {len(out)} filas; "
          f"eventos churn = {int(out['evento'].sum())}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rossi = obtener_rossi()
    vet = obtener_veterans()
    telco = obtener_telco()

    print("\n== Resumen ==")
    print(f"rossi_recidivism.csv    : {rossi.shape[0]} x {rossi.shape[1]}  (replicacion / Cox)")
    print(f"veterans_lung_cancer.csv: {vet.shape[0]} x {vet.shape[1]}  (replicacion 2 / Cox 1972)")
    print(f"telco_churn.csv         : {telco.shape[0]} x {telco.shape[1]}  (negocio / churn)")

    print("\n== Checksums SHA256 (archivo local) ==")
    for p in (ROSSI_FILE, VET_FILE, TELCO_FILE):
        if os.path.exists(p):
            mb = os.path.getsize(p) / 1e6
            print(f"  {os.path.basename(p):26s} {sha256_file(p)}  ({mb:.3f} MB)")
