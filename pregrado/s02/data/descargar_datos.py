# -*- coding: utf-8 -*-
"""
Descarga y verificacion de los datasets de la Sesion S02
(EDA + estadistica inferencial + A/B testing) del curso UPC
"Herramientas para la Ciencia de Datos".

Dos datasets:
  1) SLEEP  (replicacion del paper seminal) -- Student (1908),
     "The Probable Error of a Mean", Biometrika 6(1); datos de
     Cushny & Peebles. Estructura PAREADA: 10 sujetos x 2 farmacos.
     Fuente primaria: Rdatasets (paquete R 'datasets', dataset 'sleep').
  2) COOKIE CATS (caso de negocio, A/B test de retencion de un juego
     movil). Fuente canonica: Kaggle 'yufengsui/mobile-games-ab-testing'.
     Sin token de Kaggle (decision de curso: mirrors abiertos por sesion),
     se descarga de mirrors abiertos verificados en GitHub (primario +
     alterno byte-identico: mismo SHA256). Ademas se versiona en el repo
     una muestra <=1 MB con checksum como respaldo reproducible.

Idempotente: si el archivo ya existe y su SHA256 coincide, NO vuelve a
descargar. Funciona en local (Windows/venv) y en Google Colab (usa el
directorio de trabajo actual).

Uso:
    "<python_del_venv>" descargar_datos.py
    # o en Colab:  %run descargar_datos.py

Fecha de verificacion de fuentes: 17/07/2026.
"""
import hashlib
import io
import os
import sys

try:
    import requests
except ImportError:  # en Colab requests viene preinstalado
    requests = None

import pandas as pd

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (curso UPC S02 descargar_datos)"}

# Directorio de datos = carpeta de este script (local) o cwd (Colab).
if "google.colab" in sys.modules:
    DATA_DIR = os.getcwd()
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

SLEEP_FILE = os.path.join(DATA_DIR, "sleep.csv")
COOKIE_FILE = os.path.join(DATA_DIR, "cookie_cats.csv")
COOKIE_SAMPLE_FILE = os.path.join(DATA_DIR, "cookie_cats_muestra.csv")

# Fuentes (primaria + fallbacks)
SLEEP_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/sleep.csv"
COOKIE_MIRRORS = [
    # Mirror PRIMARIO abierto verificado (fork del proyecto DataCamp "Mobile Games
    # A/B Testing with Cookie Cats"). Esquema y n de filas identicos a Kaggle.
    "https://raw.githubusercontent.com/ryanschaub/"
    "Mobile-Games-A-B-Testing-with-Cookie-Cats/master/cookie_cats.csv",
    # Mirror ALTERNO verificado (byte-identico: mismo SHA256 que el primario,
    # verificado el 14/08/2026). Evita que un 404 del primario rompa el A/B en Colab.
    "https://raw.githubusercontent.com/yfsui/"
    "Mobile-Games-AB-Testing-Project/master/cookie_cats.csv",
]

# Checksums de referencia (verificados el 17/07/2026; mirror alterno el 14/08/2026).
SHA256_SLEEP = "5563ff4da6cf477df91551d9455f905dd0090cd82948fca985eda8a5440e548f"
SHA256_COOKIE = "5ab54d761fbddcd50de7b88e4eaf7837cba4569474f50c043a4d17ee342c46bd"
# Muestra <=1 MB versionada en el repo (respaldo reproducible con checksum fijado).
SHA256_COOKIE_SAMPLE = "83ed79d2d30477f17445d7006abf2ef93fb7329a211e4a31c6c910b15902e474"

# Esquemas esperados
SLEEP_COLS = ["rownames", "extra", "group", "ID"]   # 'rownames' = indice R
SLEEP_ROWS = 20
COOKIE_COLS = ["userid", "version", "sum_gamerounds", "retention_1", "retention_7"]
COOKIE_ROWS = 90189

# ---------------------------------------------------------------------------
# Fallback embebido de SLEEP (datos PUBLICADOS, no sinteticos)
# Valores canonicos de R 'datasets::sleep' (Cushny & Peebles, via Student 1908).
# Garantiza reproducibilidad en Colab si la fuente primaria no responde.
# ---------------------------------------------------------------------------
_SLEEP_EXTRA_G1 = [0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0]
_SLEEP_EXTRA_G2 = [1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4]


def _sleep_fallback_df() -> pd.DataFrame:
    filas = []
    idx = 1
    for g, valores in ((1, _SLEEP_EXTRA_G1), (2, _SLEEP_EXTRA_G2)):
        for sujeto, extra in enumerate(valores, start=1):
            filas.append({"rownames": idx, "extra": extra, "group": g, "ID": sujeto})
            idx += 1
    return pd.DataFrame(filas, columns=SLEEP_COLS)


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
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.content


def _verificar_esquema(df: pd.DataFrame, cols, nrows, nombre: str) -> None:
    faltan = [c for c in cols if c not in df.columns]
    if faltan:
        raise ValueError(f"[{nombre}] faltan columnas {faltan}; hay {list(df.columns)}")
    if nrows is not None and len(df) != nrows:
        raise ValueError(f"[{nombre}] se esperaban {nrows} filas y hay {len(df)}")


def _ya_valido(path: str, sha_ref: str) -> bool:
    """Idempotencia: archivo presente y con checksum esperado."""
    if not os.path.exists(path):
        return False
    if sha_ref is None:
        return True
    actual = sha256_file(path)
    if actual == sha_ref:
        return True
    print(f"  ! {os.path.basename(path)} existe pero su SHA256 no coincide "
          f"(esperado {sha_ref[:12]}..., actual {actual[:12]}...). Se re-descarga.")
    return False


# ---------------------------------------------------------------------------
# Descargas
# ---------------------------------------------------------------------------
def obtener_sleep() -> pd.DataFrame:
    print("== SLEEP (replicacion: Student 1908 / Cushny-Peebles) ==")
    if _ya_valido(SLEEP_FILE, SHA256_SLEEP):
        print(f"  = presente y verificado: {SLEEP_FILE}")
        return pd.read_csv(SLEEP_FILE)

    df = None
    try:
        raw = _fetch(SLEEP_URL)
        df = pd.read_csv(io.BytesIO(raw))
        _verificar_esquema(df, SLEEP_COLS, SLEEP_ROWS, "sleep")
        with open(SLEEP_FILE, "wb") as f:
            f.write(raw)
        print(f"  + descargado de Rdatasets -> {SLEEP_FILE}")
    except Exception as e:
        print(f"  ! fuente primaria fallo ({e!r}); uso FALLBACK embebido "
              f"(datos publicados de Cushny-Peebles, no sinteticos).")
        df = _sleep_fallback_df()
        df.to_csv(SLEEP_FILE, index=False, encoding="utf-8")
        print(f"  + escrito fallback -> {SLEEP_FILE}")

    _verificar_esquema(df, SLEEP_COLS, SLEEP_ROWS, "sleep")
    assert df["group"].nunique() == 2, "sleep debe tener 2 grupos (farmacos)"
    assert df["ID"].nunique() == 10, "sleep debe tener 10 sujetos"
    print(f"  ok: {df.shape[0]} filas x {df.shape[1]} cols; "
          f"SHA256={sha256_file(SLEEP_FILE)[:16]}...")
    return df


def obtener_cookie_cats() -> pd.DataFrame:
    print("== COOKIE CATS (negocio: A/B test de retencion) ==")
    if _ya_valido(COOKIE_FILE, SHA256_COOKIE):
        print(f"  = presente y verificado: {COOKIE_FILE}")
        return pd.read_csv(COOKIE_FILE)

    ultimo_error = None
    for url in COOKIE_MIRRORS:
        try:
            raw = _fetch(url)
            df = pd.read_csv(io.BytesIO(raw))
            _verificar_esquema(df, COOKIE_COLS, COOKIE_ROWS, "cookie_cats")
            with open(COOKIE_FILE, "wb") as f:
                f.write(raw)
            print(f"  + descargado del mirror -> {COOKIE_FILE}")
            print(f"    ({url})")
            print(f"  ok: {df.shape[0]} filas x {df.shape[1]} cols; "
                  f"SHA256={sha256_file(COOKIE_FILE)[:16]}...")
            # muestra <=1MB para vistazos rapidos / repos
            _guardar_muestra(df)
            return df
        except Exception as e:
            ultimo_error = e
            print(f"  ! mirror fallo ({e!r})")

    raise RuntimeError(
        "No se pudo obtener cookie_cats de ningun mirror. "
        "Descarga manual: Kaggle 'yufengsui/mobile-games-ab-testing' "
        "-> cookie_cats.csv en esta carpeta. Ver README_datos.md. "
        f"Ultimo error: {ultimo_error!r}"
    )


def _guardar_muestra(df: pd.DataFrame, n: int = 3000) -> None:
    # La muestra (<=1 MB) se VERSIONA en el repo con checksum fijado: es el respaldo
    # reproducible para vistazos rapidos si ningun mirror responde. Idempotente: si ya
    # existe y su SHA256 coincide con el pin, se conserva la versionada (no se re-escribe).
    if _ya_valido(COOKIE_SAMPLE_FILE, SHA256_COOKIE_SAMPLE):
        kb = os.path.getsize(COOKIE_SAMPLE_FILE) / 1024
        print(f"  = muestra versionada presente y verificada ({kb:.0f} KB) -> {COOKIE_SAMPLE_FILE}")
        return
    muestra = df.sample(n=min(n, len(df)), random_state=42).sort_index()
    muestra.to_csv(COOKIE_SAMPLE_FILE, index=False, encoding="utf-8")
    sha_new = sha256_file(COOKIE_SAMPLE_FILE)
    kb = os.path.getsize(COOKIE_SAMPLE_FILE) / 1024
    coincide = "OK" if sha_new == SHA256_COOKIE_SAMPLE else "DISTINTO (revisar version de pandas)"
    print(f"  + muestra ({len(muestra)} filas, {kb:.0f} KB) -> {COOKIE_SAMPLE_FILE}")
    print(f"    SHA256 muestra {sha_new[:16]}... vs pin {SHA256_COOKIE_SAMPLE[:16]}... [{coincide}]")


# ---------------------------------------------------------------------------
def main() -> int:
    print(f"Directorio de datos: {DATA_DIR}\n")
    sleep = obtener_sleep()
    print()
    cookie = obtener_cookie_cats()
    print("\n== Resumen ==")
    print(f"sleep       : {sleep.shape[0]} x {sleep.shape[1]}  {list(sleep.columns)}")
    print(f"cookie_cats : {cookie.shape[0]} x {cookie.shape[1]}  {list(cookie.columns)}")
    print("Verificacion de fuentes: 17/07/2026. Todo OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
