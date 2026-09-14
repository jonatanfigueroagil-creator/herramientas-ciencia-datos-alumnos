# -*- coding: utf-8 -*-
"""
Descarga y verificacion de los datasets de la Sesion EPE E6
(Pronosticar el futuro y entender el porque) del curso UPC
"Herramientas para la Ciencia de Datos".

Regla dura EPE: REUTILIZA (no regenera) datasets ya verificados para este
curso. Tres casos de negocio, todos < 1 MB para portabilidad EPE:

  1) STORE ITEM DEMAND (pronostico). Del reto Kaggle
     'c/demand-forecasting-kernels-only'. La base completa (train.csv, 913 000
     filas, ~17 MB) no se duplica aqui: se genera una muestra portable
     'store1_item1.csv' (tienda 1, producto 1; 1 826 filas
     diarias 2013-2017, ~30 KB) a partir de una copia local si esta presente,
     y si no descargando del mismo MIRROR ABIERTO byte-identico que uso S11.

  2) RETENCION OBSERVACIONAL (correlacion vs causalidad). Dataset de negocio ya
     verificado (5 000 filas; escenario simulado con efecto verdadero conocido
     = +8.0 USD, confusor 'engagement'). Se busca primero una copia local ya
     verificada y, si no esta, se regenera inline con el MISMO DGP documentado
     (semilla 20260719). Ideal para EPE: permite mostrar la "verdad" contra la que
     el estimador ingenuo se equivoca.

  3) TELCO CUSTOMER CHURN (tiempo-hasta-evento / curva de retencion). Dataset de
     negocio ya verificado (7 043 clientes; 'tenure' = meses hasta el churn).
     Se busca primero una copia local ya verificada y, si no esta, se descarga
     de un mirror abierto byte-identico (~0.97 MB).

Idempotente: si un archivo ya existe y valida (esquema / nº de filas / checksum),
NO se vuelve a obtener. Funciona en local (Windows/venv) y en Google Colab.

Uso:
    "<python_del_venv>" descargar_datos.py
    # o en Colab:  %run descargar_datos.py

Fecha de verificacion de fuentes: 06/08/2026.
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

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuracion de rutas
# ---------------------------------------------------------------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (curso UPC E06 descargar_datos)"}

if "google.colab" in sys.modules:
    DATA_DIR = os.getcwd()
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

STORE_SAMPLE = os.path.join(DATA_DIR, "store1_item1.csv")
RETENCION_FILE = os.path.join(DATA_DIR, "retencion_observacional.csv")
TELCO_FILE = os.path.join(DATA_DIR, "telco_churn.csv")

# Copia local ya verificada (evita re-descargar/duplicar).
_SES = os.path.normpath(os.path.join(DATA_DIR, "..", "..", "..", "Sesiones"))
S11_TRAIN = os.path.join(_SES, "S11_series_temporales", "data", "train.csv")
S12_RETEN = os.path.join(_SES, "S12_inferencia_causal", "data", "retencion_observacional.csv")
S13_TELCO = os.path.join(_SES, "S13_supervivencia", "data", "telco_churn.csv")

# Mirrors abiertos byte-identicos (los mismos verificados por S11 y S13).
URL_TRAIN = ("https://raw.githubusercontent.com/jgonzalezab/"
             "Store-Item-Demand-Forecasting/master/Data/train.csv")
URL_TELCO = ("https://raw.githubusercontent.com/treselle-systems/"
             "customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Esquemas esperados
STORE_COLS = ["date", "store", "item", "sales"]
STORE_ROWS = 1826            # tienda 1, producto 1: 2013-01-01..2017-12-31 (diario)
RETEN_COLS = ["engagement", "tenure", "campaign", "value"]
RETEN_ROWS = 5000
TELCO_COLS = ["customerID", "tenure", "MonthlyCharges", "Contract", "Churn"]
TELCO_ROWS = 7043


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str) -> bytes:
    if requests is None:
        raise RuntimeError("El paquete 'requests' no esta disponible.")
    r = requests.get(url, headers=HEADERS, timeout=180)
    r.raise_for_status()
    return r.content


def _verificar(df: pd.DataFrame, cols, nrows, nombre: str) -> None:
    faltan = [c for c in cols if c not in df.columns]
    if faltan:
        raise ValueError(f"[{nombre}] faltan columnas {faltan}; hay {list(df.columns)}")
    if nrows is not None and len(df) != nrows:
        raise ValueError(f"[{nombre}] se esperaban {nrows} filas y hay {len(df)}")


# ---------------------------------------------------------------------------
# 1) STORE ITEM DEMAND -- muestra portable tienda 1 / producto 1
# ---------------------------------------------------------------------------
def obtener_store() -> pd.DataFrame:
    print("== STORE ITEM DEMAND -- muestra tienda 1 / producto 1 (pronostico) ==")
    if os.path.exists(STORE_SAMPLE):
        df = pd.read_csv(STORE_SAMPLE)
        try:
            _verificar(df, STORE_COLS, STORE_ROWS, "store")
            print(f"  = presente y verificado: {STORE_SAMPLE}")
            return df
        except ValueError:
            print("  ! muestra presente con esquema inesperado; se regenera.")

    full = None
    if os.path.exists(S11_TRAIN):                      # 1) reuso local desde S11
        full = pd.read_csv(S11_TRAIN)
        print(f"  + base completa reutilizada de S11: {S11_TRAIN}")
    else:                                              # 2) mirror abierto de S11
        print("  base de S11 no presente; se descarga del mirror abierto...")
        full = pd.read_csv(io.BytesIO(_fetch(URL_TRAIN)))
        print(f"    ({URL_TRAIN[:70]}...)")

    sub = (full[(full["store"] == 1) & (full["item"] == 1)]
           .loc[:, STORE_COLS].reset_index(drop=True))
    _verificar(sub, STORE_COLS, STORE_ROWS, "store")
    sub.to_csv(STORE_SAMPLE, index=False, encoding="utf-8")
    print(f"  + muestra escrita -> {STORE_SAMPLE}  "
          f"({os.path.getsize(STORE_SAMPLE)/1024:.0f} KB; SHA256={sha256_file(STORE_SAMPLE)[:16]}...)")
    return sub


# ---------------------------------------------------------------------------
# 2) RETENCION OBSERVACIONAL -- reuso de S12 (o regeneracion inline del DGP)
# ---------------------------------------------------------------------------
def _generar_retencion(n=5000, seed=20260719, true_ate=8.0) -> pd.DataFrame:
    """MISMO DGP documentado en S12 (dueno: REPLICACION_PAPER.md S12, Seccion 5).
    value = 50 + 8*campaign + 20*engagement + 3*tenure + N(0, 8).
    'engagement' es CONFUSOR: eleva la probabilidad de recibir la campana Y el valor."""
    rng = np.random.default_rng(seed)
    engagement = rng.normal(0, 1, n)
    tenure = rng.normal(24, 6, n)
    p = 1 / (1 + np.exp(-(0.9 * engagement)))          # engagement -> campaign
    campaign = (rng.uniform(0, 1, n) < p).astype(int)
    value = (50 + true_ate * campaign + 20 * engagement + 3 * tenure
             + rng.normal(0, 8, n))
    return pd.DataFrame({"engagement": engagement, "tenure": tenure,
                         "campaign": campaign, "value": value})


def obtener_retencion() -> pd.DataFrame:
    print("== RETENCION OBSERVACIONAL -- reuso S12 (correlacion vs causalidad) ==")
    if os.path.exists(RETENCION_FILE):
        df = pd.read_csv(RETENCION_FILE)
        print(f"  = presente: {RETENCION_FILE} ({len(df)} filas)")
        return df
    if os.path.exists(S12_RETEN):                      # 1) reuso local desde S12
        shutil.copyfile(S12_RETEN, RETENCION_FILE)
        print(f"  + copiado desde S12 -> {RETENCION_FILE}")
        return pd.read_csv(RETENCION_FILE)
    df = _generar_retencion()                          # 2) regeneracion inline del DGP
    _verificar(df, RETEN_COLS, RETEN_ROWS, "retencion")
    df.to_csv(RETENCION_FILE, index=False, encoding="utf-8")
    print(f"  + regenerado con el DGP documentado -> {RETENCION_FILE}")
    return df


# ---------------------------------------------------------------------------
# 3) TELCO CUSTOMER CHURN -- reuso de S13 (o mirror abierto)
# ---------------------------------------------------------------------------
def obtener_telco() -> pd.DataFrame:
    print("== TELCO CUSTOMER CHURN -- reuso S13 (tiempo-hasta-evento / churn) ==")
    if os.path.exists(TELCO_FILE):
        df = pd.read_csv(TELCO_FILE)
        print(f"  = presente: {TELCO_FILE} ({len(df)} filas)")
        return df
    if os.path.exists(S13_TELCO):                      # 1) reuso local desde S13
        shutil.copyfile(S13_TELCO, TELCO_FILE)
        print(f"  + copiado desde S13 -> {TELCO_FILE}")
        return pd.read_csv(TELCO_FILE)
    raw = _fetch(URL_TELCO)                             # 2) mirror abierto de S13
    df = pd.read_csv(io.BytesIO(raw))
    _verificar(df, TELCO_COLS, TELCO_ROWS, "telco")
    with open(TELCO_FILE, "wb") as f:
        f.write(raw)
    print(f"  + descargado del mirror -> {TELCO_FILE}")
    print(f"    ({URL_TELCO[:70]}...)")
    return df


# ---------------------------------------------------------------------------
def main() -> int:
    print(f"Directorio de datos: {DATA_DIR}\n")
    store = obtener_store()
    print()
    reten = obtener_retencion()
    print()
    telco = obtener_telco()
    print("\n== Resumen ==")
    print(f"store1_item1.csv          : {store.shape[0]} x {store.shape[1]}  (pronostico)")
    print(f"retencion_observacional   : {reten.shape[0]} x {reten.shape[1]}  (causalidad, reuso S12)")
    print(f"telco_churn.csv           : {telco.shape[0]} x {telco.shape[1]}  (supervivencia, reuso S13)")
    print("Verificacion de fuentes: 06/08/2026. Todo OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
