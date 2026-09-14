# -*- coding: utf-8 -*-
"""
Descarga y verificacion de los datos de la Sesion S04
(Regresion Lineal Multiple: diagnostico, dummies, interacciones) del curso UPC
"Herramientas para la Ciencia de Datos".

DATASET: Ames Housing (De Cock, 2011). Un solo conjunto de casas de Ames, Iowa
(2006-2010) que cumple los DOS roles de la FICHA:

  1) REPLICACION DEL PAPER (anclaje de los numeros) -> "AmesHousing.csv"
     El AmesHousing COMPLETO: 2930 filas x 82 columnas, exactamente el archivo
     que De Cock publico con su articulo. Es el unico que reproduce sus
     observaciones canonicas (correlacion SalePrice ~ Gr Liv Area y las 5
     ventas atipicas con Gr Liv Area > 4000 sq ft que recomienda remover).
       - Fuente primaria: Journal of Statistics Education, hosting del propio
         autor: http://jse.amstat.org/v19n3/decock/AmesHousing.txt (tab-sep).
       - Fallback: Rdatasets, paquete R 'openintro', dataset 'ames' (2930 x 82,
         mismas variables con otros nombres -> se normalizan a los canonicos).
     Cita: De Cock, D. (2011). "Ames, Iowa: Alternative to the Boston Housing
     Data as an End of Semester Regression Project". Journal of Statistics
     Education 19(3). https://doi.org/10.1080/10691898.2011.11889627

  2) NEGOCIO / LABORATORIO (nombres estilo Kaggle) -> "AmesHousing_kaggle.csv"
     El subconjunto de la competencia Kaggle "House Prices - Advanced
     Regression Techniques" (train): 1460 filas x 81 columnas, con los nombres
     sin espacios (GrLivArea, OverallQual, ...). Es un ~50% aleatorio del
     conjunto completo; sirve para el caso de negocio (zona x tamano).
       - Carga primaria: sklearn.datasets.fetch_openml(data_id=42165)
         (OpenML espeja el train de Kaggle; Kaggle exige login -> sin token,
         decision del curso 17/07/2026).

Ambos archivos pesan < 1.5 MB -> se conservan en OneDrive (regla: solo los
>25 MB se excluyen).

Idempotente: si el CSV ya existe y su esquema (filas x columnas + columnas
clave) es correcto, NO vuelve a descargar. Funciona en local (Windows/venv) y
en Google Colab (usa el directorio de trabajo actual).

Uso:
    "<python_del_venv>" descargar_datos.py            # descarga + verifica esquema
    "<python_del_venv>" descargar_datos.py --smoke     # ademas corre el smoke de viabilidad
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
HEADERS = {"User-Agent": "Mozilla/5.0 (curso UPC S04 descargar_datos)"}

if "google.colab" in sys.modules:
    DATA_DIR = os.getcwd()
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

FULL_FILE = os.path.join(DATA_DIR, "AmesHousing.csv")          # 2930 x 82 (paper)
KAGGLE_FILE = os.path.join(DATA_DIR, "AmesHousing_kaggle.csv")  # 1460 x 81 (negocio)

# --- Fuentes del conjunto COMPLETO (paper) ---
JSE_URL = "http://jse.amstat.org/v19n3/decock/AmesHousing.txt"  # tab-separado
# SHA256 de los BYTES CRUDOS del .txt de JSE (integridad de la fuente primaria).
SHA256_JSE_RAW = "6cfe6cb525ba437de428653a1040e2aed7d696640bf75203786a6d7a0e67cfcc"
FALLBACK_FULL_URL = (
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/ames.csv"
)

# --- Fuente del subconjunto Kaggle (negocio) ---
OPENML_DATA_ID = 42165

# Columnas clave (nombres canonicos De Cock, CON espacios) que deben existir
# en el conjunto completo para poder replicar y hacer el laboratorio.
FULL_KEY_COLS = [
    "Order", "PID", "SalePrice", "Gr Liv Area", "Overall Qual", "Year Built",
    "Neighborhood", "Garage Cars", "Garage Area", "Total Bsmt SF",
    "Sale Condition",
]
FULL_ROWS, FULL_COLS = 2930, 82

# Columnas clave del subconjunto Kaggle (nombres SIN espacios).
KAGGLE_KEY_COLS = [
    "Id", "SalePrice", "GrLivArea", "OverallQual", "YearBuilt", "Neighborhood",
]
KAGGLE_ROWS, KAGGLE_COLS = 1460, 81


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
    r = requests.get(url, headers=HEADERS, timeout=90)
    r.raise_for_status()
    return r.content


def _verificar_esquema(df, key_cols, nrows, ncols, nombre):
    faltan = [c for c in key_cols if c not in df.columns]
    if faltan:
        raise ValueError(
            f"[{nombre}] faltan columnas clave {faltan}; "
            f"hay {df.shape[1]} columnas"
        )
    if nrows is not None and len(df) != nrows:
        raise ValueError(f"[{nombre}] se esperaban {nrows} filas y hay {len(df)}")
    if ncols is not None and df.shape[1] != ncols:
        raise ValueError(
            f"[{nombre}] se esperaban {ncols} columnas y hay {df.shape[1]}"
        )


def _csv_valido(path, key_cols, nrows, ncols) -> bool:
    """Idempotencia: CSV presente y con esquema correcto."""
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path, low_memory=False)
        _verificar_esquema(df, key_cols, nrows, ncols, os.path.basename(path))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ! {os.path.basename(path)} existe pero no valida ({e}); se re-descarga.")
        return False


# ---------------------------------------------------------------------------
# Normalizacion de nombres del fallback (Rdatasets openintro/ames)
# a los nombres canonicos de De Cock (CON espacios).
# ---------------------------------------------------------------------------
def _canon_openintro(col: str) -> str:
    especiales = {
        "area": "Gr Liv Area",
        "price": "SalePrice",
        "X1st.Flr.SF": "1st Flr SF",
        "X2nd.Flr.SF": "2nd Flr SF",
        "X3Ssn.Porch": "3Ssn Porch",
    }
    if col in especiales:
        return especiales[col]
    return col.replace(".", " ")


# ---------------------------------------------------------------------------
# Descarga del conjunto COMPLETO (replicacion del paper)
# ---------------------------------------------------------------------------
def obtener_ames_full() -> pd.DataFrame:
    print("== AMES COMPLETO 2930x82 (replicacion: De Cock 2011) ==")
    if _csv_valido(FULL_FILE, FULL_KEY_COLS, FULL_ROWS, FULL_COLS):
        print(f"  = presente y verificado: {FULL_FILE}")
        return pd.read_csv(FULL_FILE, low_memory=False)

    # Fuente primaria: JSE (tab-separado, nombres canonicos con espacios).
    try:
        raw = _fetch(JSE_URL)
        sha = sha256_bytes(raw)
        aviso = "" if sha == SHA256_JSE_RAW else "  [SHA256 de fuente NUEVO]"
        df = pd.read_csv(io.BytesIO(raw), sep="\t", low_memory=False)
        _verificar_esquema(df, FULL_KEY_COLS, FULL_ROWS, FULL_COLS, "JSE AmesHousing")
        df.to_csv(FULL_FILE, index=False, encoding="utf-8")
        print(f"  + descargado de JSE (fuente primaria del paper){aviso}")
        print(f"    {JSE_URL}")
        print(f"    SHA256(txt crudo)={sha[:16]}...  -> {FULL_FILE}")
        return df
    except Exception as e:  # noqa: BLE001
        print(f"  ! JSE fallo ({e!r}); intentando fallback Rdatasets openintro...")

    # Fallback: Rdatasets openintro/ames -> normalizar nombres.
    raw = _fetch(FALLBACK_FULL_URL)
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    if "rownames" in df.columns:
        df = df.drop(columns=["rownames"])
    df = df.rename(columns={c: _canon_openintro(c) for c in df.columns})
    _verificar_esquema(df, FULL_KEY_COLS, FULL_ROWS, FULL_COLS, "openintro/ames")
    df.to_csv(FULL_FILE, index=False, encoding="utf-8")
    print(f"  + descargado del FALLBACK Rdatasets openintro/ames (nombres normalizados)")
    print(f"    {FALLBACK_FULL_URL}")
    print(f"    -> {FULL_FILE}")
    return df


# ---------------------------------------------------------------------------
# Descarga del subconjunto KAGGLE (negocio)
# ---------------------------------------------------------------------------
def obtener_ames_kaggle() -> pd.DataFrame:
    print("== AMES KAGGLE 1460x81 (negocio: House Prices train) ==")
    if _csv_valido(KAGGLE_FILE, KAGGLE_KEY_COLS, KAGGLE_ROWS, KAGGLE_COLS):
        print(f"  = presente y verificado: {KAGGLE_FILE}")
        return pd.read_csv(KAGGLE_FILE, low_memory=False)

    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True)
    df = d.frame
    _verificar_esquema(df, KAGGLE_KEY_COLS, KAGGLE_ROWS, KAGGLE_COLS, "openml 42165")
    df.to_csv(KAGGLE_FILE, index=False, encoding="utf-8")
    print(f"  + cargado via fetch_openml(data_id={OPENML_DATA_ID}) -> {KAGGLE_FILE}")
    return df


# ---------------------------------------------------------------------------
# SMOKE de VIABILIDAD de la replica (reporta valores OBSERVADOS, NO define targets)
# ---------------------------------------------------------------------------
def smoke():
    import numpy as np
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score

    full = obtener_ames_full()
    GLA, SP, OQ, YB = "Gr Liv Area", "SalePrice", "Overall Qual", "Year Built"
    GC, GA = "Garage Cars", "Garage Area"

    print("\n== SMOKE 1: correlacion y atipicos (conjunto completo 2930) ==")
    r_all = full[SP].corr(full[GLA])
    n_atip = int((full[GLA] > 4000).sum())
    clean = full[full[GLA] <= 4000].copy()
    r_clean = clean[SP].corr(clean[GLA])
    print(f"  corr SalePrice~Gr Liv Area (todas): {r_all:.4f}")
    print(f"  casas con Gr Liv Area > 4000 (De Cock recomienda remover): {n_atip}")
    print(f"  corr sin las {n_atip} atipicas (n={len(clean)}): {r_clean:.4f}")

    print("\n== SMOKE 2: OLS multiple con statsmodels (limpio) ==")
    print("   SalePrice ~ Gr Liv Area + Overall Qual")
    X = sm.add_constant(clean[[GLA, OQ]])
    y = clean[SP]
    res = sm.OLS(y, X).fit()
    print(f"  R2={res.rsquared:.4f}  R2_adj={res.rsquared_adj:.4f}  n={int(res.nobs)}")
    print(f"  coef Gr Liv Area={res.params[GLA]:.4f} (USD/sq ft)  "
          f"Overall Qual={res.params[OQ]:.2f} (USD/punto)")
    print(f"  todos los p-value < 0.001: {bool((res.pvalues < 0.001).all())}")
    res_hc3 = sm.OLS(y, X).fit(cov_type="HC3")
    print(f"  SE robusto HC3 (White) Gr Liv Area={res_hc3.bse[GLA]:.4f} "
          f"vs OLS clasico={res.bse[GLA]:.4f}  (indicio de heterocedasticidad)")

    print("\n== SMOKE 3: VIF de 3 predictores ==")
    sub = clean[[GLA, OQ, YB]].dropna()
    Xv = sm.add_constant(sub)
    for i, c in enumerate(Xv.columns):
        if c == "const":
            continue
        print(f"  VIF[{c}] = {variance_inflation_factor(Xv.values, i):.3f}")
    print("  (colinealidad ilustrativa Garage Cars vs Garage Area:)")
    sub2 = clean[[GC, GA, GLA]].dropna()
    Xv2 = sm.add_constant(sub2)
    for i, c in enumerate(Xv2.columns):
        if c == "const":
            continue
        print(f"  VIF[{c}] = {variance_inflation_factor(Xv2.values, i):.3f}")

    print("\n== SMOKE 4: k-fold CV (5-fold, R2) con sklearn ==")
    print("   SalePrice ~ Gr Liv Area + Overall Qual + Year Built")
    cv_df = clean[[GLA, OQ, YB, SP]].dropna()
    Xc = cv_df[[GLA, OQ, YB]].values
    yc = cv_df[SP].values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(LinearRegression(), Xc, yc, cv=kf, scoring="r2")
    print(f"  R2 por fold: {np.round(scores, 4)}")
    print(f"  R2 medio CV: {scores.mean():.4f} (+/- {scores.std():.4f})")

    print("\n== SMOKE 5: contraste con subconjunto Kaggle (negocio) ==")
    kag = obtener_ames_kaggle()
    print(f"  corr SalePrice~GrLivArea (1460): "
          f"{kag['SalePrice'].corr(kag['GrLivArea']):.4f}  |  "
          f">4000: {int((kag['GrLivArea'] > 4000).sum())}")
    print("\n  (Los targets con tolerancia los fija investigador-tema en "
          "la ficha de réplica del paper; aqui solo se OBSERVAN.)")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    print(f"Directorio de datos: {DATA_DIR}\n")
    full = obtener_ames_full()
    print()
    kag = obtener_ames_kaggle()
    print("\n== Resumen ==")
    print(f"AmesHousing.csv        (paper)   : {full.shape[0]} x {full.shape[1]}")
    print(f"AmesHousing_kaggle.csv (negocio) : {kag.shape[0]} x {kag.shape[1]}")
    for p in (FULL_FILE, KAGGLE_FILE):
        if os.path.exists(p):
            print(f"  SHA256({os.path.basename(p)}) = {sha256_file(p)}")
    if "--smoke" in argv:
        smoke()
    print("\nVerificacion de fuentes: 18/07/2026. Todo OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
