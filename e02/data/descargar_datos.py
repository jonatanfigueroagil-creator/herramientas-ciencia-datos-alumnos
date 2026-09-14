# -*- coding: utf-8 -*-
"""
Descarga y verificacion de los datasets de la Sesion EPE E2
(Regresion lineal para predecir) del curso UPC "Herramientas para la Ciencia de Datos".

Regla dura EPE: REUTILIZA, no regenera datasets ya verificados para este curso:

  1) ADVERTISING (regresion SIMPLE) -- inversion en TV, radio y prensa vs.
     ventas de un producto en 200 mercados. Dataset clasico de ISLR
     (James, Witten, Hastie & Tibshirani). Fuente canonica del silabo EPE:
     www.statlearning.com/s/Advertising.csv. En local se busca primero una
     copia ya verificada del curso; si no esta, se descarga del mismo mirror.
     Pesa < 25 KB -> se conserva en la carpeta.

  2) AMES HOUSING (regresion MULTIPLE) -- precios de vivienda en Ames, Iowa
     (De Cock, 2011). Se usa el subconjunto de negocio (train de la competencia
     Kaggle "House Prices", 1460 casas x 81 columnas, nombres sin espacios:
     GrLivArea, OverallQual, ...). Silabo EPE: Kaggle o via
     fetch_openml(data_id=42165). En local se busca primero una copia ya
     verificada del curso; si no esta, se carga con
     sklearn.datasets.fetch_openml(42165). Pesa < 1.5 MB -> se conserva en la
     carpeta.

Idempotente: si un archivo ya existe y su esquema (columnas clave + nº de filas)
es correcto, NO vuelve a obtenerlo. Funciona en local (Windows/venv) y en Google
Colab (usa el directorio de trabajo actual).

Uso:
    "<python_del_venv>" descargar_datos.py
    # o en Colab:  %run descargar_datos.py

Fecha de verificacion de fuentes: 06/08/2026.
"""
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
# Configuracion
# ---------------------------------------------------------------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (curso UPC E02 descargar_datos)"}

# Directorio de datos = carpeta de este script (local) o cwd (Colab).
if "google.colab" in sys.modules:
    DATA_DIR = os.getcwd()
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

ADV_FILE = os.path.join(DATA_DIR, "Advertising.csv")
AMES_FILE = os.path.join(DATA_DIR, "AmesHousing_kaggle.csv")

# Reuso local del material ya cerrado de S03 y S04 (evita re-descargar).
_S03_DATA = os.path.normpath(os.path.join(
    DATA_DIR, "..", "..", "..", "Sesiones", "S03_ols", "data"))
_S04_DATA = os.path.normpath(os.path.join(
    DATA_DIR, "..", "..", "..", "Sesiones", "S04_regresion_multiple", "data"))
S03_ADV = os.path.join(_S03_DATA, "Advertising.csv")
S04_AMES = os.path.join(_S04_DATA, "AmesHousing_kaggle.csv")

# Fuentes (mismas que usaron S03 / S04)
ADV_URLS = [
    "https://www.statlearning.com/s/Advertising.csv",
    # Fallback: mirror ISLR ampliamente usado (mismo esquema y nº de filas).
    "https://raw.githubusercontent.com/nguyen-toan/ISLR/master/dataset/Advertising.csv",
]
OPENML_AMES_ID = 42165  # OpenML espeja el train de Kaggle "House Prices"

# Esquemas esperados
ADV_COLS = ["TV", "radio", "newspaper", "sales"]
ADV_ROWS = 200
AMES_KEY_COLS = ["Id", "SalePrice", "GrLivArea", "OverallQual", "YearBuilt",
                 "Neighborhood", "GarageCars", "TotalBsmtSF"]
AMES_ROWS, AMES_COLS = 1460, 81


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _fetch(url: str) -> bytes:
    if requests is None:
        raise RuntimeError("requests no disponible")
    r = requests.get(url, headers=HEADERS, timeout=90)
    r.raise_for_status()
    return r.content


def _verificar_esquema(df, cols, nrows, nombre, ncols=None):
    faltan = [c for c in cols if c not in df.columns]
    if faltan:
        raise ValueError(f"[{nombre}] faltan columnas {faltan}; hay {list(df.columns)[:8]}...")
    if nrows is not None and len(df) != nrows:
        raise ValueError(f"[{nombre}] se esperaban {nrows} filas y hay {len(df)}")
    if ncols is not None and df.shape[1] != ncols:
        raise ValueError(f"[{nombre}] se esperaban {ncols} columnas y hay {df.shape[1]}")


def _csv_valido(path, cols, nrows, nombre, ncols=None) -> bool:
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path, low_memory=False)
        _verificar_esquema(df, cols, nrows, nombre, ncols)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ! {os.path.basename(path)} existe pero no valida ({e}); se re-obtiene.")
        return False


# ---------------------------------------------------------------------------
# ADVERTISING (regresion simple) -- reuso de S03
# ---------------------------------------------------------------------------
def obtener_advertising() -> pd.DataFrame:
    print("== ADVERTISING (regresion simple: ventas ~ inversion) ==")
    if _csv_valido(ADV_FILE, ADV_COLS, ADV_ROWS, "advertising"):
        print(f"  = presente y verificado: {ADV_FILE}")
        return pd.read_csv(ADV_FILE)
    # 1) reuso local desde S03 (evita re-descargar)
    if _csv_valido(S03_ADV, ADV_COLS, ADV_ROWS, "advertising-S03"):
        shutil.copyfile(S03_ADV, ADV_FILE)
        print(f"  + copiado desde S03 -> {ADV_FILE}")
        return pd.read_csv(ADV_FILE)
    # 2) descarga del mismo mirror que uso S03
    ultimo_error = None
    for url in ADV_URLS:
        try:
            raw = _fetch(url)
            df = pd.read_csv(io.BytesIO(raw))
            _verificar_esquema(df, ADV_COLS, ADV_ROWS, "advertising")
            with open(ADV_FILE, "wb") as f:
                f.write(raw)
            print(f"  + descargado -> {ADV_FILE}")
            print(f"    ({url})")
            return df
        except Exception as e:  # noqa: BLE001
            ultimo_error = e
            print(f"  ! fuente fallo ({e!r})")
    raise RuntimeError(
        "No se pudo obtener Advertising. Descarga manual: "
        "www.statlearning.com/s/Advertising.csv en esta carpeta. "
        f"Ultimo error: {ultimo_error!r}")


# ---------------------------------------------------------------------------
# AMES HOUSING (regresion multiple) -- reuso de S04
# ---------------------------------------------------------------------------
def obtener_ames() -> pd.DataFrame:
    print("== AMES HOUSING (regresion multiple: precio ~ caracteristicas) ==")
    if _csv_valido(AMES_FILE, AMES_KEY_COLS, AMES_ROWS, "ames", AMES_COLS):
        print(f"  = presente y verificado: {AMES_FILE}")
        return pd.read_csv(AMES_FILE, low_memory=False)
    # 1) reuso local desde S04 (evita re-descargar)
    if _csv_valido(S04_AMES, AMES_KEY_COLS, AMES_ROWS, "ames-S04", AMES_COLS):
        shutil.copyfile(S04_AMES, AMES_FILE)
        print(f"  + copiado desde S04 -> {AMES_FILE}")
        return pd.read_csv(AMES_FILE, low_memory=False)
    # 2) carga via fetch_openml (mismo id que uso S04)
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=OPENML_AMES_ID, as_frame=True)
    df = d.frame
    _verificar_esquema(df, AMES_KEY_COLS, AMES_ROWS, "openml 42165", AMES_COLS)
    df.to_csv(AMES_FILE, index=False, encoding="utf-8")
    print(f"  + cargado via fetch_openml(data_id={OPENML_AMES_ID}) -> {AMES_FILE}")
    return df


# ---------------------------------------------------------------------------
def main() -> int:
    print(f"Directorio de datos: {DATA_DIR}\n")
    adv = obtener_advertising()
    print()
    ames = obtener_ames()
    print("\n== Resumen ==")
    print(f"Advertising            : {adv.shape[0]} x {adv.shape[1]}  {ADV_COLS}")
    print(f"AmesHousing_kaggle.csv : {ames.shape[0]} x {ames.shape[1]}")
    print("Verificacion de fuentes: 06/08/2026. Todo OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
