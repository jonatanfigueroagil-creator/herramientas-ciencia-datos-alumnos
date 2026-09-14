# -*- coding: utf-8 -*-
"""
Descarga y verificacion de los datasets de la Sesion S03
(Regresion Lineal Simple -- OLS) del curso UPC
"Herramientas para la Ciencia de Datos".

Tres fuentes de datos:
  1) GALTON FAMILIES (replicacion del paper seminal) -- Galton, F. (1886),
     "Regression Towards Mediocrity in Hereditary Stature",
     Journal of the Anthropological Institute 15, 246-263. Es el estudio que
     dio origen al termino "regresion". 934 hijos de 205 familias, con la
     altura del padre, de la madre y la "mid-parent height".
     Fuente primaria: Rdatasets (paquete R 'HistData', dataset
     'GaltonFamilies'), reconstruido del cuaderno original por Hanley (2004).
  2) ADVERTISING (caso de negocio del laboratorio) -- inversion en TV, radio
     y prensa vs. ventas de un producto en 200 mercados. Dataset clasico de
     ISLR (James, Witten, Hastie & Tibshirani, "An Introduction to
     Statistical Learning"). Fuente primaria: www.statlearning.com.
  3) ANSCOMBE (apoyo conceptual) -- cuarteto de Anscombe (1973). NO requiere
     descarga externa: viene INTEGRADO en seaborn (`load_dataset("anscombe")`).
     Este script solo verifica que carga (4 datasets I-IV, 11 puntos c/u).

Idempotente: si el archivo ya existe y su SHA256 coincide, NO vuelve a
descargar. Funciona en local (Windows/venv) y en Google Colab (usa el
directorio de trabajo actual). Ambos CSV pesan < 25 MB -> se conservan.

Uso:
    "<python_del_venv>" descargar_datos.py           # descarga + verifica esquema
    "<python_del_venv>" descargar_datos.py --smoke    # ademas corre el smoke OLS
    # o en Colab:  %run descargar_datos.py

Fecha de verificacion de fuentes: 18/07/2026.
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
HEADERS = {"User-Agent": "Mozilla/5.0 (curso UPC S03 descargar_datos)"}

# Directorio de datos = carpeta de este script (local) o cwd (Colab).
if "google.colab" in sys.modules:
    DATA_DIR = os.getcwd()
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

GALTON_FILE = os.path.join(DATA_DIR, "GaltonFamilies.csv")
ADV_FILE = os.path.join(DATA_DIR, "Advertising.csv")

# Fuentes (primaria + fallbacks). El primero es el enlace del silabo.
GALTON_URLS = [
    "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/GaltonFamilies.csv",
]
ADV_URLS = [
    "https://www.statlearning.com/s/Advertising.csv",
    # Fallback: mirror ISLR ampliamente usado (mismo esquema y n de filas).
    "https://raw.githubusercontent.com/nguyen-toan/ISLR/master/dataset/Advertising.csv",
]

# Checksums de referencia (descarga real verificada el 18/07/2026).
SHA256_GALTON = "b1fc91e4d4b9604661967dcc62ff1055eaa3020c4d5c42d1f9d588f74b9b3317"
SHA256_ADV = "5d88819733db0d929716e9c8fa87f20fe1ec12a9ba2feb43ed7e69e585ace3ad"

# Esquemas esperados
GALTON_COLS = ["rownames", "family", "father", "mother", "midparentHeight",
               "children", "childNum", "gender", "childHeight"]
GALTON_ROWS = 934
# Advertising: el CSV de statlearning trae una columna indice sin nombre
# ("Unnamed: 0") ademas de las 4 variables del silabo.
ADV_COLS = ["TV", "radio", "newspaper", "sales"]
ADV_ROWS = 200


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


def _descargar(urls, destino, cols, nrows, sha_ref, nombre):
    ultimo_error = None
    for url in urls:
        try:
            raw = _fetch(url)
            df = pd.read_csv(io.BytesIO(raw))
            _verificar_esquema(df, cols, nrows, nombre)
            with open(destino, "wb") as f:
                f.write(raw)
            print(f"  + descargado -> {destino}")
            print(f"    ({url})")
            actual = sha256_file(destino)
            aviso = "" if (sha_ref is None or actual == sha_ref) else "  [SHA256 NUEVO]"
            print(f"  ok: {df.shape[0]} filas x {df.shape[1]} cols; "
                  f"SHA256={actual[:16]}...{aviso}")
            return df
        except Exception as e:  # noqa: BLE001
            ultimo_error = e
            print(f"  ! fuente fallo ({e!r})")
    raise RuntimeError(
        f"No se pudo obtener {nombre} de ninguna fuente. "
        f"Ultimo error: {ultimo_error!r}"
    )


# ---------------------------------------------------------------------------
# Descargas
# ---------------------------------------------------------------------------
def obtener_galton() -> pd.DataFrame:
    print("== GALTON FAMILIES (replicacion: Galton 1886) ==")
    if _ya_valido(GALTON_FILE, SHA256_GALTON):
        print(f"  = presente y verificado: {GALTON_FILE}")
        return pd.read_csv(GALTON_FILE)
    df = _descargar(GALTON_URLS, GALTON_FILE, GALTON_COLS, GALTON_ROWS,
                    SHA256_GALTON, "GaltonFamilies")
    assert df["gender"].str.lower().isin(["male", "female"]).all(), \
        "gender debe ser male/female"
    assert df["family"].nunique() == 205, "Galton reporto 205 familias"
    return df


def obtener_advertising() -> pd.DataFrame:
    print("== ADVERTISING (negocio: ventas ~ inversion publicitaria) ==")
    if _ya_valido(ADV_FILE, SHA256_ADV):
        print(f"  = presente y verificado: {ADV_FILE}")
        return pd.read_csv(ADV_FILE)
    df = _descargar(ADV_URLS, ADV_FILE, ADV_COLS, ADV_ROWS, SHA256_ADV,
                    "Advertising")
    return df


def verificar_anscombe():
    """El cuarteto de Anscombe viene INTEGRADO en seaborn (no se descarga)."""
    print("== ANSCOMBE (apoyo conceptual: seaborn integrado) ==")
    try:
        import seaborn as sns
    except ImportError as e:
        raise RuntimeError(f"seaborn no disponible: {e!r}")
    ans = sns.load_dataset("anscombe")
    datasets = sorted(ans["dataset"].unique())
    if datasets != ["I", "II", "III", "IV"]:
        raise ValueError(f"Anscombe: se esperaban 4 datasets I-IV; hay {datasets}")
    conteo = ans.groupby("dataset").size()
    print(f"  ok: {ans.shape[0]} filas x {ans.shape[1]} cols; "
          f"datasets {datasets} ({list(conteo.values)} puntos c/u). "
          f"Integrado en seaborn, sin descarga externa.")
    return ans


# ---------------------------------------------------------------------------
# Smoke test de VIABILIDAD de la replica (OLS childHeight ~ midparentHeight)
# ---------------------------------------------------------------------------
def smoke_ols(df: pd.DataFrame = None):
    """Ajusta OLS con statsmodels y sklearn; confirma que ambos importan y
    corren y reporta pendiente/intercepto/r/R2 OBSERVADOS. NO define targets."""
    import numpy as np
    import statsmodels.api as sm
    from sklearn.linear_model import LinearRegression

    if df is None:
        df = obtener_galton()
    print("\n== SMOKE OLS: childHeight ~ midparentHeight (crudo, ambos sexos) ==")
    x = df["midparentHeight"].to_numpy()
    y = df["childHeight"].to_numpy()

    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit()
    b0, b1 = res.params
    r = float(np.corrcoef(x, y)[0, 1])
    print(f"  statsmodels : intercepto={b0:.4f}  pendiente={b1:.4f}  "
          f"R2={res.rsquared:.4f}  r={r:.4f}")

    lr = LinearRegression().fit(x.reshape(-1, 1), y)
    print(f"  sklearn     : intercepto={lr.intercept_:.4f}  "
          f"pendiente={lr.coef_[0]:.4f}  R2={lr.score(x.reshape(-1,1), y):.4f}")

    # Variante: alturas de las hijas x1.08 ('transmutacion' de Galton/Hanley).
    print("\n== SMOKE OLS: hijas x1.08 (transmutacion de Galton) ==")
    mask_f = df["gender"].str.lower().str.startswith("f")
    y2 = df["childHeight"].to_numpy().copy()
    y2[mask_f.to_numpy()] = y2[mask_f.to_numpy()] * 1.08
    res2 = sm.OLS(y2, X).fit()
    r2 = float(np.corrcoef(x, y2)[0, 1])
    print(f"  statsmodels : intercepto={res2.params[0]:.4f}  "
          f"pendiente={res2.params[1]:.4f}  R2={res2.rsquared:.4f}  r={r2:.4f}")
    print("\n  (Los targets con tolerancia los fija investigador-tema en "
          "la ficha de réplica del paper; aqui solo se OBSERVAN.)")
    return {
        "crudo": {"b0": float(b0), "b1": float(b1), "r": r,
                  "R2": float(res.rsquared)},
        "hijas_x1.08": {"b0": float(res2.params[0]), "b1": float(res2.params[1]),
                        "r": r2, "R2": float(res2.rsquared)},
    }


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    print(f"Directorio de datos: {DATA_DIR}\n")
    galton = obtener_galton()
    print()
    adv = obtener_advertising()
    print()
    ans = verificar_anscombe()
    print("\n== Resumen ==")
    print(f"GaltonFamilies : {galton.shape[0]} x {galton.shape[1]}  {list(galton.columns)}")
    print(f"Advertising    : {adv.shape[0]} x {adv.shape[1]}  {list(adv.columns)}")
    print(f"Anscombe       : {ans.shape[0]} x {ans.shape[1]}  (seaborn integrado)")
    if "--smoke" in argv:
        smoke_ols(galton)
    print("\nVerificacion de fuentes: 18/07/2026. Todo OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
