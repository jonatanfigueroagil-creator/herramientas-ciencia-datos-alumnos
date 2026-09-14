# -*- coding: utf-8 -*-
"""
Descarga y verificacion de los datos de la Sesion EPE E4
(Segmentar clientes y encontrar patrones) del curso UPC
"Herramientas para la Ciencia de Datos".

Esta sesion trabaja con **Online Retail II** (UCI id 502). Aqui NO se descarga
el completo (~45.6 MB): se usan dos MUESTRAS ya preparadas (<=1 MB cada una),
cada una segun su unidad de analisis:

  1) ONLINE RETAIL II - por CLIENTE (segmentacion RFM). Muestra de 140 clientes
     con toda su historia de compra (18 125 filas), de modo que Recencia,
     Frecuencia y Monto quedan exactas por cliente. Guardada aqui como
     'online_retail_II_rfm_muestra.csv'.

  2) ONLINE RETAIL II - por FACTURA/TICKET (reglas de asociacion / market
     basket). Muestra de 800 facturas con todas sus lineas (16 979 filas) y con
     'Description' (el nombre del producto legible que se necesita para leer
     reglas), guardada como 'online_retail_baskets_muestra.csv'.

Idempotente: si un archivo ya existe y su SHA256 coincide, NO vuelve a
obtenerlo. Funciona en local (Windows/venv) y en Google Colab. En local copia
una version local ya verificada si esta presente; si no, la descarga del mirror
abierto del repositorio del curso (GitHub raw). En Colab descarga del mirror.

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

import pandas as pd

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (curso UPC E04 descargar_datos)"}

if "google.colab" in sys.modules:
    DATA_DIR = os.getcwd()
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

RFM_FILE = os.path.join(DATA_DIR, "online_retail_II_rfm_muestra.csv")
BASKETS_FILE = os.path.join(DATA_DIR, "online_retail_baskets_muestra.csv")

# Copia local ya verificada (si existe), para evitar re-descargar.
_CACHE_RFM_DIR = os.path.normpath(os.path.join(
    DATA_DIR, "..", "..", "..", "Sesiones", "S07_clustering_anomalias", "data"))
_CACHE_BASKETS_DIR = os.path.normpath(os.path.join(
    DATA_DIR, "..", "..", "..", "Sesiones", "S08_reglas_asociacion", "data"))
CACHE_RFM_CSV = os.path.join(_CACHE_RFM_DIR, "online_retail_II_muestra.csv")
CACHE_BASKETS_CSV = os.path.join(_CACHE_BASKETS_DIR, "online_retail_baskets_muestra.csv")

# Mirror abierto (repositorio del curso en GitHub) para Colab / respaldo portable.
_RAW = ("https://raw.githubusercontent.com/jonatanfigueroagil-creator/"
        "Herramientas-de-Ciencias-de-Datos/master/Sesiones_EPE/"
        "E04_segmentar_patrones/data/")
RFM_MIRROR = _RAW + "online_retail_II_rfm_muestra.csv"
BASKETS_MIRROR = _RAW + "online_retail_baskets_muestra.csv"

# Checksums de referencia (verificados 06/08/2026).
SHA256_RFM = "96b678af3bebf0d6277e0e101c7fca0745b37c6d080a2158df0adc3bc58e0aef"
SHA256_BASKETS = "a3745101de08f44a8fc67167e9075cb12427a776b53e7d0cdc503a78ddae2760"

# Esquemas esperados
RFM_COLS = ["Invoice", "InvoiceDate", "Quantity", "Price", "Customer ID", "Country"]
RFM_ROWS = 18125
BASKETS_COLS = ["Invoice", "StockCode", "Description", "Country"]
BASKETS_ROWS = 16979


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
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.content


def _verificar_esquema(df: pd.DataFrame, cols, nrows, nombre: str) -> None:
    faltan = [c for c in cols if c not in df.columns]
    if faltan:
        raise ValueError(f"[{nombre}] faltan columnas {faltan}; hay {list(df.columns)}")
    if nrows is not None and len(df) != nrows:
        raise ValueError(f"[{nombre}] se esperaban {nrows} filas y hay {len(df)}")


def _valido(path: str, sha: str) -> bool:
    return os.path.exists(path) and sha256_file(path) == sha


def _obtener(destino, sha, cols, nrows, cache_local, mirror, nombre):
    """Idempotente: copia local ya verificada si esta; si no, descarga del mirror del curso."""
    print(f"== {nombre} ==")
    if _valido(destino, sha):
        print(f"  = presente y verificado: {destino}")
        return pd.read_csv(destino)
    # 1) copia local ya verificada (evita re-descargar)
    if os.path.exists(cache_local) and sha256_file(cache_local) == sha:
        shutil.copyfile(cache_local, destino)
        print(f"  + copiada desde cache local -> {destino}")
        return pd.read_csv(destino)
    # 2) descarga del mirror abierto del repositorio del curso
    try:
        raw = _fetch(mirror)
        df = pd.read_csv(io.BytesIO(raw))
        _verificar_esquema(df, cols, nrows, nombre)
        with open(destino, "wb") as f:
            f.write(raw)
        print(f"  + descargada del mirror del curso -> {destino}")
        print(f"  ok: {df.shape[0]} filas x {df.shape[1]} cols; "
              f"SHA256={sha256_file(destino)[:16]}...")
        return df
    except Exception as e:
        raise RuntimeError(
            f"No se pudo obtener {nombre}. Copiar manualmente una muestra valida "
            f"a esta carpeta, o descargar Online Retail II de "
            f"UCI id 502 (https://archive.ics.uci.edu/dataset/502/online+retail+ii) "
            f"y preparar la muestra. Ultimo error: {e!r}")


def obtener_rfm() -> pd.DataFrame:
    return _obtener(RFM_FILE, SHA256_RFM, RFM_COLS, RFM_ROWS, CACHE_RFM_CSV, RFM_MIRROR,
                    "ONLINE RETAIL II - por cliente (segmentacion RFM)")


def obtener_baskets() -> pd.DataFrame:
    return _obtener(BASKETS_FILE, SHA256_BASKETS, BASKETS_COLS, BASKETS_ROWS,
                    CACHE_BASKETS_CSV, BASKETS_MIRROR,
                    "ONLINE RETAIL II - por factura (market basket)")


# ---------------------------------------------------------------------------
def main() -> int:
    print(f"Directorio de datos: {DATA_DIR}\n")
    rfm = obtener_rfm()
    print()
    baskets = obtener_baskets()
    print("\n== Resumen ==")
    print(f"RFM (por cliente)   : {rfm.shape[0]} x {rfm.shape[1]}  "
          f"({rfm['Customer ID'].nunique()} clientes)")
    print(f"Canastas (por ticket): {baskets.shape[0]} x {baskets.shape[1]}  "
          f"({baskets['Invoice'].nunique()} facturas)")
    print("Verificacion de fuentes: 06/08/2026. Todo OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
