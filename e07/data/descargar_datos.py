# -*- coding: utf-8 -*-
"""
E7 (EPE) — Sistemas de recomendacion y proyecto integrador — descarga/verificacion.

Este script verifica si los datasets ya estan disponibles localmente (con el esquema
esperado) y, si faltan, los descarga de la fuente abierta o de un mirror del
repositorio del curso.

Produce en esta carpeta (Sesiones_EPE/E07_recomendacion/data/):
  1) ml100k_ratings.csv  — MovieLens 100k: 100.000 valoraciones (943 usuarios x
                           1682 peliculas), escala 1-5. Base del recomendador.
  2) ml100k_movies.csv   — metadatos de las 1682 peliculas (titulo + 19 flags de
                           genero) para el filtrado por contenido.
  3) online_retail_reco_muestra.csv — NEGOCIO (feedback implicito). Muestra por
                           cliente de Online Retail II (UCI 502), <=1 MB.

Reglas del curso:
  * MovieLens 100k = SUSTITUTO CANONICO del Netflix Prize (retirado). En EPE se usa
    a nivel de intuicion (sin replica formal en esta sesion).
  * ratings/movies CSV (<25 MB) SI pueden quedar en el repo -> base completa en disco.
  * El COMPLETO de Online Retail II (~45,6 MB) NUNCA en OneDrive: solo la muestra.
  * Verificacion REAL: se carga programaticamente y se validan filas x columnas.

Uso:
    "<python_del_venv>" descargar_datos.py
    # o en Colab:  %run descargar_datos.py

Fecha de verificacion de fuentes: 06/08/2026.
"""
from __future__ import annotations

import hashlib
import io
import os
import shutil
import sys
import zipfile

try:
    import requests
except Exception:  # en Colab requests viene preinstalado
    requests = None

import pandas as pd

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
if "google.colab" in sys.modules:
    DATA_DIR = os.getcwd()
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Copia local ya verificada (evita re-descargar).
_ORIGEN_LOCAL = os.path.normpath(os.path.join(
    DATA_DIR, "..", "..", "..", "Sesiones", "S14_recomendacion", "data"))

# Mirror del repo del curso (respaldo para Colab).
RAW = ("https://raw.githubusercontent.com/jonatanfigueroagil-creator/"
       "Herramientas-de-Ciencias-de-Datos/master/Sesiones/S14_recomendacion/data/")

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML100K_GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

FILES = {
    "ml100k_ratings.csv": ["user_id", "item_id", "rating", "timestamp"],
    "ml100k_movies.csv":  ["movie_id", "title", "release_date"] + ML100K_GENRES,
    "online_retail_reco_muestra.csv": [
        "Invoice", "InvoiceDate", "StockCode", "Description",
        "Quantity", "Price", "Customer ID", "Country"],
}
EXPECTED = {
    "ml100k_ratings.csv": (100000, 943, 1682),
    "ml100k_movies.csv":  (1682,),
    "online_retail_reco_muestra.csv": (80,),  # nº de clientes
}


# ---------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str) -> bytes:
    if requests is None:
        raise RuntimeError("requests no disponible.")
    r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0 (curso UPC E07)"})
    r.raise_for_status()
    return r.content


def _schema_ok(path: str, cols: list[str]) -> bool:
    if not os.path.exists(path):
        return False
    try:
        head = pd.read_csv(path, nrows=5)
    except Exception:
        return False
    return list(head.columns) == cols


def _validar(nombre: str, path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    exp = EXPECTED[nombre]
    if nombre == "ml100k_ratings.csv":
        assert len(df) == exp[0] and df.user_id.nunique() == exp[1] and df.item_id.nunique() == exp[2], \
            f"[{nombre}] conteos inesperados: {len(df)}, {df.user_id.nunique()}, {df.item_id.nunique()}"
    elif nombre == "ml100k_movies.csv":
        assert len(df) == exp[0], f"[{nombre}] se esperaban {exp[0]} filas y hay {len(df)}"
    else:
        assert df["Customer ID"].nunique() == exp[0], \
            f"[{nombre}] se esperaban {exp[0]} clientes y hay {df['Customer ID'].nunique()}"
    return df


# ---------------------------------------------------------------------------
def _descargar_movielens() -> None:
    """Descarga ml-100k.zip y escribe ratings + movies (si faltan)."""
    print(f"  ... descargando MovieLens 100k de {ML100K_URL} (~4,9 MB)")
    z = zipfile.ZipFile(io.BytesIO(_fetch(ML100K_URL)))
    rt = pd.read_csv(io.BytesIO(z.read("ml-100k/u.data")), sep="\t",
                     names=["user_id", "item_id", "rating", "timestamp"], encoding="latin-1")
    rt.to_csv(os.path.join(DATA_DIR, "ml100k_ratings.csv"), index=False, encoding="utf-8")
    item_cols = (["movie_id", "title", "release_date", "video_release", "imdb_url"] + ML100K_GENRES)
    mv = pd.read_csv(io.BytesIO(z.read("ml-100k/u.item")), sep="|",
                     names=item_cols, encoding="latin-1")
    mv = mv[["movie_id", "title", "release_date"] + ML100K_GENRES]
    mv.to_csv(os.path.join(DATA_DIR, "ml100k_movies.csv"), index=False, encoding="utf-8")


def obtener(nombre: str) -> pd.DataFrame:
    dst = os.path.join(DATA_DIR, nombre)
    cols = FILES[nombre]

    # 1) presente y con esquema OK
    if _schema_ok(dst, cols):
        try:
            df = _validar(nombre, dst)
            print(f"  = presente y verificado: {nombre}  ({len(df)} filas)")
            return df
        except AssertionError as e:
            print(f"  ! {e}; se re-obtiene.")

    # 2) copia local ya verificada
    src = os.path.join(_ORIGEN_LOCAL, nombre)
    if _schema_ok(src, cols):
        shutil.copyfile(src, dst)
        print(f"  + copiado desde caché local -> {nombre}")
        return _validar(nombre, dst)

    # 3) descarga de la fuente abierta / mirror del repo
    if nombre.startswith("ml100k"):
        _descargar_movielens()
    else:
        raw = _fetch(RAW + nombre)
        with open(dst, "wb") as f:
            f.write(raw)
        print(f"  + descargado del repo del curso -> {nombre}")
    return _validar(nombre, dst)


# ---------------------------------------------------------------------------
def main() -> int:
    print(f"Directorio de datos: {DATA_DIR}")
    print("Verificando datos locales...\n")
    dfs = {n: obtener(n) for n in FILES}

    print("\n== Resumen ==")
    rt = dfs["ml100k_ratings.csv"]
    sp = 1 - len(rt) / (rt.user_id.nunique() * rt.item_id.nunique())
    print(f"MovieLens ratings : {rt.shape[0]} x {rt.shape[1]}  "
          f"({rt.user_id.nunique()} usuarios x {rt.item_id.nunique()} peliculas, "
          f"dispersion {sp:.1%})")
    print(f"MovieLens movies  : {dfs['ml100k_movies.csv'].shape[0]} peliculas (19 flags de genero)")
    rr = dfs["online_retail_reco_muestra.csv"]
    print(f"Online Retail (mx): {rr.shape[0]} x {rr.shape[1]}  "
          f"({rr['Customer ID'].nunique()} clientes, {rr['StockCode'].nunique()} productos)")

    print("\n== Checksums SHA256 (archivos locales) ==")
    for n in FILES:
        p = os.path.join(DATA_DIR, n)
        if os.path.exists(p):
            print(f"  {n:34s} {os.path.getsize(p)/1e6:6.3f} MB  {sha256_file(p)}")
    print("\nListo. Datos de E7 (EPE) verificados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
