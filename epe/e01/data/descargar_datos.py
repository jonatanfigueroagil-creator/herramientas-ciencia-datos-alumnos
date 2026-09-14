# -*- coding: utf-8 -*-
"""
Descarga y verificacion de los datasets de la Sesion EPE E1
(Fundamentos: CRISP-DM + exploracion de datos + A/B testing) del curso UPC
"Herramientas para la Ciencia de Datos".

Dos casos de negocio (rebaja de complejidad EPE: datasets pequenos y directos):

  1) MALL CUSTOMERS  (exploracion / EDA). Perfiles de 200 clientes de un centro
     comercial: sexo, edad, ingreso anual y puntaje de gasto. Fuente canonica:
     Kaggle 'vjchoudhary7/customer-segmentation-tutorial-in-python'. Sin token de
     Kaggle (decision de curso: mirrors abiertos por sesion) se descarga de un
     mirror abierto verificado en GitHub. Se normaliza la columna de sexo a
     'Gender' (algunos mirrors la nombran 'Genre').

  2) COOKIE CATS  (experimento A/B de retencion de un juego movil). Dataset ya
     verificado para este curso:
       - la muestra 'cookie_cats_muestra.csv' (3 000 filas) se copia a esta
         carpeta para un vistazo rapido y portable;
       - la base completa 'cookie_cats.csv' (90 189 filas, 2.7 MB < 25 MB) es la
         que sostiene el analisis A/B; en local se busca primero una copia ya
         verificada del curso y, si no esta, se descarga de un mirror abierto.
         En Colab se descarga del mirror.

Idempotente: si un archivo ya existe y su SHA256 coincide, NO vuelve a
descargar. Funciona en local (Windows/venv) y en Google Colab (usa el
directorio de trabajo actual).

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
HEADERS = {"User-Agent": "Mozilla/5.0 (curso UPC E01 descargar_datos)"}

# Directorio de datos = carpeta de este script (local) o cwd (Colab).
if "google.colab" in sys.modules:
    DATA_DIR = os.getcwd()
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

MALL_FILE = os.path.join(DATA_DIR, "Mall_Customers.csv")
COOKIE_FILE = os.path.join(DATA_DIR, "cookie_cats.csv")
COOKIE_SAMPLE_FILE = os.path.join(DATA_DIR, "cookie_cats_muestra.csv")

# Reuso local del material de S02 (evita re-descargar la base completa).
_S02_DATA = os.path.normpath(os.path.join(
    DATA_DIR, "..", "..", "..", "Sesiones", "S02_eda_inferencia_ab", "data"))
S02_COOKIE = os.path.join(_S02_DATA, "cookie_cats.csv")
S02_COOKIE_SAMPLE = os.path.join(_S02_DATA, "cookie_cats_muestra.csv")

# Fuentes (primaria + fallback)
MALL_MIRRORS = [
    # Mirror abierto verificado (columna de sexo ya nombrada 'Gender').
    "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/"
    "master/Datasets/Mall_Customers.csv",
    # Fallback (columna 'Genre'; se renombra a 'Gender').
    "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/"
    "Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/"
    "Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv",
]
COOKIE_MIRRORS = [
    # Mismo mirror abierto verificado que uso S02 (fork del proyecto DataCamp
    # "Mobile Games A/B Testing with Cookie Cats"). Esquema y n identicos a Kaggle.
    "https://raw.githubusercontent.com/ryanschaub/"
    "Mobile-Games-A-B-Testing-with-Cookie-Cats/master/cookie_cats.csv",
]

# Checksum de referencia de la base completa de Cookie Cats (verificado 06/08/2026).
# Mall Customers se valida por esquema (200 filas x 5 columnas) tras normalizar la
# columna de sexo a 'Gender'; su checksum se registra en diccionario_datos.md.
SHA256_COOKIE = "5ab54d761fbddcd50de7b88e4eaf7837cba4569474f50c043a4d17ee342c46bd"

# Esquemas esperados
MALL_COLS = ["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
MALL_ROWS = 200
COOKIE_COLS = ["userid", "version", "sum_gamerounds", "retention_1", "retention_7"]
COOKIE_ROWS = 90189
COOKIE_SAMPLE_ROWS = 3000


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


def _cookie_valido(path: str) -> bool:
    """Idempotencia por checksum para la base completa de Cookie Cats."""
    if not os.path.exists(path):
        return False
    actual = sha256_file(path)
    if actual == SHA256_COOKIE:
        return True
    print(f"  ! {os.path.basename(path)} existe pero su SHA256 no coincide; se re-obtiene.")
    return False


# ---------------------------------------------------------------------------
# MALL CUSTOMERS (EDA)
# ---------------------------------------------------------------------------
def obtener_mall() -> pd.DataFrame:
    print("== MALL CUSTOMERS (exploracion / EDA) ==")
    if os.path.exists(MALL_FILE):
        df = pd.read_csv(MALL_FILE)
        try:
            _verificar_esquema(df, MALL_COLS, MALL_ROWS, "mall")
            print(f"  = presente y verificado: {MALL_FILE}")
            return df
        except ValueError:
            print("  ! archivo presente con esquema inesperado; se re-descarga.")

    ultimo_error = None
    for url in MALL_MIRRORS:
        try:
            raw = _fetch(url)
            df = pd.read_csv(io.BytesIO(raw))
            # Normalizar el nombre de la columna de sexo a 'Gender'.
            if "Genre" in df.columns and "Gender" not in df.columns:
                df = df.rename(columns={"Genre": "Gender"})
            _verificar_esquema(df, MALL_COLS, MALL_ROWS, "mall")
            df = df[MALL_COLS]
            df.to_csv(MALL_FILE, index=False, encoding="utf-8")
            print(f"  + descargado del mirror -> {MALL_FILE}")
            print(f"    ({url[:70]}...)")
            print(f"  ok: {df.shape[0]} filas x {df.shape[1]} cols; "
                  f"SHA256={sha256_file(MALL_FILE)[:16]}...")
            return df
        except Exception as e:
            ultimo_error = e
            print(f"  ! mirror fallo ({e!r})")
    raise RuntimeError(
        "No se pudo obtener Mall Customers de ningun mirror. Descarga manual: "
        "Kaggle 'vjchoudhary7/customer-segmentation-tutorial-in-python' -> "
        f"Mall_Customers.csv en esta carpeta. Ultimo error: {ultimo_error!r}")


# ---------------------------------------------------------------------------
# COOKIE CATS (A/B) -- muestra (reuso S02) + base completa
# ---------------------------------------------------------------------------
def obtener_cookie_muestra() -> pd.DataFrame:
    print("== COOKIE CATS -- muestra (reuso de S02, 3 000 filas) ==")
    if os.path.exists(COOKIE_SAMPLE_FILE):
        df = pd.read_csv(COOKIE_SAMPLE_FILE)
        print(f"  = presente: {COOKIE_SAMPLE_FILE} ({len(df)} filas)")
        return df
    # 1) reusar la muestra ya generada por S02
    if os.path.exists(S02_COOKIE_SAMPLE):
        shutil.copyfile(S02_COOKIE_SAMPLE, COOKIE_SAMPLE_FILE)
        print(f"  + copiada desde S02 -> {COOKIE_SAMPLE_FILE}")
        return pd.read_csv(COOKIE_SAMPLE_FILE)
    # 2) si no esta, derivarla de la base completa
    base = obtener_cookie_completa()
    muestra = base.sample(n=COOKIE_SAMPLE_ROWS, random_state=42).sort_index()
    muestra.to_csv(COOKIE_SAMPLE_FILE, index=False, encoding="utf-8")
    print(f"  + muestra derivada de la base completa -> {COOKIE_SAMPLE_FILE}")
    return muestra


def obtener_cookie_completa() -> pd.DataFrame:
    print("== COOKIE CATS -- base completa (analisis A/B; 90 189 filas) ==")
    if _cookie_valido(COOKIE_FILE):
        print(f"  = presente y verificado: {COOKIE_FILE}")
        return pd.read_csv(COOKIE_FILE)
    # 1) reuso local desde S02 (evita re-descargar)
    if os.path.exists(S02_COOKIE) and sha256_file(S02_COOKIE) == SHA256_COOKIE:
        shutil.copyfile(S02_COOKIE, COOKIE_FILE)
        print(f"  + copiada desde S02 -> {COOKIE_FILE}")
        return pd.read_csv(COOKIE_FILE)
    # 2) descarga del mirror abierto
    ultimo_error = None
    for url in COOKIE_MIRRORS:
        try:
            raw = _fetch(url)
            df = pd.read_csv(io.BytesIO(raw))
            _verificar_esquema(df, COOKIE_COLS, COOKIE_ROWS, "cookie_cats")
            with open(COOKIE_FILE, "wb") as f:
                f.write(raw)
            print(f"  + descargada del mirror -> {COOKIE_FILE}")
            print(f"  ok: {df.shape[0]} filas x {df.shape[1]} cols; "
                  f"SHA256={sha256_file(COOKIE_FILE)[:16]}...")
            return df
        except Exception as e:
            ultimo_error = e
            print(f"  ! mirror fallo ({e!r})")
    raise RuntimeError(
        "No se pudo obtener cookie_cats de ningun mirror. Descarga manual: "
        "Kaggle 'yufengsui/mobile-games-ab-testing' -> cookie_cats.csv aqui. "
        f"Ultimo error: {ultimo_error!r}")


# ---------------------------------------------------------------------------
def main() -> int:
    print(f"Directorio de datos: {DATA_DIR}\n")
    mall = obtener_mall()
    print()
    muestra = obtener_cookie_muestra()
    print()
    cookie = obtener_cookie_completa()
    print("\n== Resumen ==")
    print(f"Mall_Customers      : {mall.shape[0]} x {mall.shape[1]}  {list(mall.columns)}")
    print(f"cookie_cats_muestra : {muestra.shape[0]} x {muestra.shape[1]}")
    print(f"cookie_cats (full)  : {cookie.shape[0]} x {cookie.shape[1]}")
    print("Verificacion de fuentes: 06/08/2026. Todo OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
