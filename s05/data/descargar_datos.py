# -*- coding: utf-8 -*-
"""
Descarga y verificacion de los datos de la Sesion S05 del curso UPC
"Herramientas para la Ciencia de Datos"
(Regularizacion: RIDGE, LASSO, Elastic Net + regresion no lineal).

Dos datasets, dos roles la ficha de la sesióncion 4):

  1) REPLICACION DEL PAPER  ->  "prostate.csv"
     Prostate cancer (Stamey et al., 1989), el conjunto canonico de
     "The Elements of Statistical Learning" (Hastie, Tibshirani, Friedman)
     usado en el cap. 3 (Tabla 3.3) y en Tibshirani (1996) para ilustrar
     OLS / Best-subset / Ridge / Lasso / PCR / PLS.  97 hombres a punto de
     una prostatectomia radical; se predice el log del antigeno prostatico
     especifico (lpsa) a partir de medidas clinicas.  Trae la columna 'train'
     (67 T / 30 F) con la particion EXACTA de ESL -> imprescindible para
     reproducir sus numeros.  La ESTANDARIZACION de predictores es clave para
     Ridge/Lasso (ver nota de convencion en el smoke).
       - Fuente del silabo: https://hastie.su.domains/ElemStatLearn/datasets/prostate.data
         -> da HTTP-403 a scripts (bloqueo anti-bot).
       - Fuente REAL usada (mirror abierto, descarga validada): repositorio
         GitHub 'empathy87/The-Elements-of-Statistical-Learning-Python-Notebooks',
         archivo 'data/Prostate Cancer.txt' (CSV con cabecera, mismos datos).
     Cita: Stamey, T.A., Kabalin, J.N., McNeal, J.E., et al. (1989).
     "Prostate specific antigen in the diagnosis and treatment of
     adenocarcinoma of the prostate. II. Radical prostatectomy treated
     patients." The Journal of Urology 141(5), 1076-1083.
     Replica: Tibshirani, R. (1996). "Regression Shrinkage and Selection
     via the Lasso." JRSS-B 58(1), 267-288.  /  Hastie, Tibshirani &
     Friedman (2009), ESL, cap. 3, Tabla 3.3.

  2) NEGOCIO / ALTA DIMENSION  ->  "communities.csv"
     Communities and Crime (Redmond, 2009), UCI id 183.  1994 comunidades
     de EE. UU. descritas por ~100 predictores socioeconomicos (censo 1990)
     y policiales (LEMAS 1990), todos normalizados a [0,1].  La variable
     objetivo es 'ViolentCrimesPerPop' (crimenes violentos por 100 000 hab.,
     tambien normalizada).  Es el caso de alta dimension del laboratorio:
     muchos predictores correlacionados y con faltantes -> ideal para
     comparar OLS vs. Ridge/Lasso/Elastic Net.
       - Fuente REAL usada: UCI ML Repository (mirror clasico de descarga
         directa) 'communities.data' + 'communities.names'.
       - Fallback documentado: sklearn.datasets.fetch_openml (dataset
         "us_crime"), mismo conjunto espejado.
     Cita: Redmond, M. (2009). Communities and Crime Data Set. UCI Machine
     Learning Repository. https://doi.org/10.24432/C53W3X
     Los datos combinan censo 1990 de EE. UU., LEMAS 1990 y UCR 1995 del FBI.

Faltantes: en 'communities.data' se codifican con '?' -> se convierten a NaN
al guardar el CSV.  Muchas columnas policiales (bloque Lemas*/Polic*) tienen
~84% de faltantes; se documentan en diccionario_datos.md.

Ambos archivos pesan < 1.2 MB -> se conservan en OneDrive (la regla de
exclusion aplica solo a >25 MB).

Idempotente: si el CSV ya existe y su esquema (filas x columnas + columnas
clave) es correcto, NO vuelve a descargar.  Funciona en local (Windows/venv)
y en Google Colab (usa el directorio de trabajo actual).

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

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (curso UPC S05 descargar_datos)"}

if "google.colab" in sys.modules:
    DATA_DIR = os.getcwd()
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

PROSTATE_FILE = os.path.join(DATA_DIR, "prostate.csv")        # 97 x 11 (paper)
COMM_FILE = os.path.join(DATA_DIR, "communities.csv")         # 1994 x 128 (negocio)
COMM_NAMES_FILE = os.path.join(DATA_DIR, "communities.names")  # metadatos UCI

# --- Fuentes: PROSTATE ---
# Silabo (403 a scripts): https://hastie.su.domains/ElemStatLearn/datasets/prostate.data
PROSTATE_URL = (
    "https://raw.githubusercontent.com/empathy87/"
    "The-Elements-of-Statistical-Learning-Python-Notebooks/master/"
    "data/Prostate%20Cancer.txt"
)
# SHA256 de los BYTES CRUDOS del mirror (integridad de la fuente).
SHA256_PROSTATE_RAW = (
    "009f17321518c1aa6dd3b3f6739cb13ec7482e1eb86f647e7f58e045341671f6"
)

# --- Fuentes: COMMUNITIES AND CRIME (UCI id 183) ---
COMM_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "communities/communities.data"
)
COMM_NAMES_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "communities/communities.names"
)
SHA256_COMM_RAW = (
    "09e0b5c07eae24c1efab19b2edee05e160e7f5743b6f31e31eec3d73624da2ea"
)
OPENML_COMM_NAME = "us_crime"  # fallback (mismo conjunto en OpenML)

# communities.data NO trae cabecera; el orden de columnas es el de los
# 128 @attribute de communities.names (embebidos aqui para robustez).
COMM_COLS = [
    # --- 5 columnas NO predictivas ---
    "state", "county", "community", "communityname", "fold",
    # --- 122 predictores (socioeconomicos + policiales, todos en [0,1]) ---
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
    # --- variable objetivo ---
    "ViolentCrimesPerPop",
]

# Esquemas esperados
PROSTATE_KEY_COLS = [
    "lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason",
    "pgg45", "lpsa", "train",
]
PROSTATE_ROWS, PROSTATE_COLS = 97, 11  # 10 predictores/target + 'id'
PROSTATE_N_TRAIN, PROSTATE_N_TEST = 67, 30

COMM_KEY_COLS = ["state", "communityname", "fold", "ViolentCrimesPerPop"]
COMM_ROWS, COMM_COLS_N = 1994, 128


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
# PROSTATE (replicacion del paper)
# ---------------------------------------------------------------------------
def obtener_prostate() -> pd.DataFrame:
    print("== PROSTATE 97x11 (replicacion: Stamey 1989 / ESL Tabla 3.3) ==")
    if _csv_valido(PROSTATE_FILE, PROSTATE_KEY_COLS, PROSTATE_ROWS, PROSTATE_COLS):
        df = pd.read_csv(PROSTATE_FILE)
        print(f"  = presente y verificado: {PROSTATE_FILE}")
        return df

    raw = _fetch(PROSTATE_URL)
    sha = sha256_bytes(raw)
    aviso = "" if sha == SHA256_PROSTATE_RAW else "  [SHA256 de fuente NUEVO]"
    df = pd.read_csv(io.BytesIO(raw))
    # El mirror nombra 'id' a la primera columna (indice 1..97 del original).
    if "id" not in df.columns and df.columns[0].lower().startswith("unnamed"):
        df = df.rename(columns={df.columns[0]: "id"})
    _verificar_esquema(df, PROSTATE_KEY_COLS, PROSTATE_ROWS, PROSTATE_COLS,
                       "empathy87/Prostate")
    # 'train' debe ser T/F con la particion 67/30 de ESL.
    vc = df["train"].astype(str).str.upper().value_counts().to_dict()
    n_tr, n_te = vc.get("T", 0), vc.get("F", 0)
    if (n_tr, n_te) != (PROSTATE_N_TRAIN, PROSTATE_N_TEST):
        raise ValueError(
            f"[Prostate] particion train esperada {PROSTATE_N_TRAIN}/"
            f"{PROSTATE_N_TEST}, observada {n_tr}/{n_te}"
        )
    df.to_csv(PROSTATE_FILE, index=False, encoding="utf-8")
    print(f"  + descargado del mirror empathy87/ESL{aviso}")
    print(f"    {PROSTATE_URL}")
    print(f"    train={n_tr}  test={n_te}  SHA256(fuente)={sha[:16]}...  -> {PROSTATE_FILE}")
    return df


# ---------------------------------------------------------------------------
# COMMUNITIES AND CRIME (negocio / alta dimension)
# ---------------------------------------------------------------------------
def _parsear_communities(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), header=None, names=COMM_COLS,
                     na_values="?", low_memory=False)
    _verificar_esquema(df, COMM_KEY_COLS, COMM_ROWS, COMM_COLS_N,
                       "UCI communities")
    return df


def obtener_communities() -> pd.DataFrame:
    print("== COMMUNITIES AND CRIME 1994x128 (negocio: UCI 183, Redmond 2009) ==")
    if _csv_valido(COMM_FILE, COMM_KEY_COLS, COMM_ROWS, COMM_COLS_N):
        df = pd.read_csv(COMM_FILE, low_memory=False)
        print(f"  = presente y verificado: {COMM_FILE}")
        return df

    # Fuente primaria: UCI.
    try:
        raw = _fetch(COMM_DATA_URL)
        sha = sha256_bytes(raw)
        aviso = "" if sha == SHA256_COMM_RAW else "  [SHA256 de fuente NUEVO]"
        df = _parsear_communities(raw)
        df.to_csv(COMM_FILE, index=False, encoding="utf-8")
        # metadatos UCI (best-effort, no bloqueante)
        try:
            with open(COMM_NAMES_FILE, "wb") as f:
                f.write(_fetch(COMM_NAMES_URL))
        except Exception:  # noqa: BLE001
            pass
        n_na = int(df["ViolentCrimesPerPop"].isna().sum())
        print(f"  + descargado de UCI (fuente primaria){aviso}")
        print(f"    {COMM_DATA_URL}")
        print(f"    faltantes en objetivo={n_na}  SHA256(fuente)={sha[:16]}...  -> {COMM_FILE}")
        return df
    except Exception as e:  # noqa: BLE001
        print(f"  ! UCI fallo ({e!r}); intentando fallback OpenML '{OPENML_COMM_NAME}'...")

    # Fallback: OpenML.
    from sklearn.datasets import fetch_openml
    d = fetch_openml(name=OPENML_COMM_NAME, as_frame=True)
    df = d.frame
    # OpenML puede nombrar el objetivo distinto; normalizar si hace falta.
    if "ViolentCrimesPerPop" not in df.columns and d.target is not None:
        df = df.copy()
        df["ViolentCrimesPerPop"] = d.target.values
    if len(df) != COMM_ROWS:
        raise ValueError(
            f"[OpenML {OPENML_COMM_NAME}] filas {len(df)} != {COMM_ROWS}"
        )
    df.to_csv(COMM_FILE, index=False, encoding="utf-8")
    print(f"  + cargado del FALLBACK OpenML '{OPENML_COMM_NAME}' -> {COMM_FILE}")
    return df


# ---------------------------------------------------------------------------
# SMOKE de VIABILIDAD (reporta valores OBSERVADOS, NO define targets)
# ---------------------------------------------------------------------------
def smoke():
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import (
        LinearRegression, Ridge, Lasso, ElasticNetCV,
    )
    from sklearn.metrics import mean_squared_error

    print("\n== SMOKE 1: replica ESL Tabla 3.3 sobre 'prostate' ==")
    df = obtener_prostate()
    preds = ["lcavol", "lweight", "age", "lbph", "svi", "lcp", "gleason", "pgg45"]
    is_train = df["train"].astype(str).str.upper().eq("T")
    X_all = df[preds].values.astype(float)
    y_all = df["lpsa"].values.astype(float)
    ytr = y_all[is_train.values]
    yte = y_all[~is_train.values]
    print(f"  n_train={len(ytr)}  n_test={len(yte)}  p={len(preds)}")

    # CONVENCION ESL: los predictores se estandarizan a varianza unitaria sobre
    # las 97 observaciones (NO solo el train).  Es lo que reproduce la TABLA de
    # coeficientes del libro.  Con estandarizacion train-only el error de test
    # es el MISMO (0.521) pero el coeficiente de lcavol sale ~0.711 en vez de
    # ~0.68 (ver nota en diccionario_datos.md; el target lo fija investigador-tema).
    sc = StandardScaler().fit(X_all)
    Z = sc.transform(X_all)
    Xtr, Xte = Z[is_train.values], Z[~is_train.values]

    # --- OLS ---
    ols = LinearRegression().fit(Xtr, ytr)
    coef_ols = dict(zip(preds, ols.coef_))
    mse_ols = mean_squared_error(yte, ols.predict(Xte))
    base_te = mean_squared_error(yte, np.full_like(yte, ytr.mean()))
    print(f"  [OLS]  coef(lcavol)={coef_ols['lcavol']:.4f}  intercepto={ols.intercept_:.4f}")
    print(f"         (ESL Tabla 3.3: lcavol 0.68, intercepto 2.46, test error 0.521)")
    print(f"         MSE_test={mse_ols:.4f}   MSE_test base (media)={base_te:.4f}")

    # --- Ridge ---
    ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
    mse_ridge = mean_squared_error(yte, ridge.predict(Xte))
    print(f"  [Ridge a=1.0]  coef(lcavol)={ridge.coef_[0]:.4f}  MSE_test={mse_ridge:.4f}")

    # --- Lasso (varios lambda para ver como anula variables) ---
    print("  [Lasso] variables distintas de cero por alpha:")
    for a in (0.01, 0.05, 0.1, 0.2, 0.3):
        la = Lasso(alpha=a, max_iter=100000).fit(Xtr, ytr)
        nz = int(np.sum(np.abs(la.coef_) > 1e-8))
        anuladas = [p for p, c in zip(preds, la.coef_) if abs(c) <= 1e-8]
        mse = mean_squared_error(yte, la.predict(Xte))
        print(f"      alpha={a:<5} no-cero={nz}/8  anula={anuladas}  "
              f"coef(lcavol)={la.coef_[0]:.4f}  MSE_test={mse:.4f}")

    # --- Elastic Net por CV ---
    enet = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1.0], cv=5,
                        max_iter=100000, random_state=0).fit(Xtr, ytr)
    nz_en = int(np.sum(np.abs(enet.coef_) > 1e-8))
    mse_en = mean_squared_error(yte, enet.predict(Xte))
    print(f"  [ElasticNetCV]  l1_ratio*={enet.l1_ratio_}  alpha*={enet.alpha_:.4f}")
    print(f"         no-cero={nz_en}/8  coef(lcavol)={enet.coef_[0]:.4f}  MSE_test={mse_en:.4f}")

    print("\n== SMOKE 2: pyGAM (regresion no lineal) sobre 'prostate' ==")
    try:
        from pygam import LinearGAM, s
        # spline sobre 2 predictores continuos (lcavol, lweight)
        Xg = df[["lcavol", "lweight"]].values.astype(float)
        yg = df["lpsa"].values.astype(float)
        gam = LinearGAM(s(0) + s(1)).fit(Xg, yg)
        r2 = gam.statistics_.get("pseudo_r2", {}).get("explained_deviance", float("nan"))
        print(f"  pyGAM OK: LinearGAM(s(lcavol)+s(lweight)) ajustado sobre n={len(yg)}")
        print(f"     pseudo-R2 (explained deviance)={r2:.4f}  "
              f"edof={gam.statistics_.get('edof', float('nan')):.2f}")
    except Exception as e:  # noqa: BLE001
        print(f"  pyGAM FALLA: {type(e).__name__}: {e}")
        print("     Fallback sugerido: splines via statsmodels/patsy "
              "(dmatrix('bs(lcavol, df=4)')) + statsmodels.GLM/OLS.")

    print("\n== SMOKE 3: alta dimension (communities) — OLS vs Lasso ==")
    dc = obtener_communities()
    no_pred = ["state", "county", "community", "communityname", "fold",
               "ViolentCrimesPerPop"]
    y = dc["ViolentCrimesPerPop"]
    X = dc.drop(columns=no_pred).apply(pd.to_numeric, errors="coerce")
    n_cols_na = int((X.isna().sum() > 0).sum())
    X = X.fillna(X.mean())  # imputacion simple por la media solo para el smoke
    print(f"  X={X.shape}  y={y.shape}  columnas-predictoras con faltantes: {n_cols_na}")
    from sklearn.model_selection import train_test_split
    Xtr2, Xte2, ytr2, yte2 = train_test_split(X.values, y.values,
                                              test_size=0.3, random_state=0)
    sc2 = StandardScaler().fit(Xtr2)
    Xtr2s, Xte2s = sc2.transform(Xtr2), sc2.transform(Xte2)
    ols2 = LinearRegression().fit(Xtr2s, ytr2)
    lcv = ElasticNetCV(l1_ratio=1.0, cv=5, max_iter=100000, random_state=0).fit(Xtr2s, ytr2)
    nz2 = int(np.sum(np.abs(lcv.coef_) > 1e-8))
    print(f"  [OLS]   MSE_test={mean_squared_error(yte2, ols2.predict(Xte2s)):.5f}")
    print(f"  [Lasso] alpha*={lcv.alpha_:.5f}  no-cero={nz2}/{X.shape[1]}  "
          f"MSE_test={mean_squared_error(yte2, lcv.predict(Xte2s)):.5f}")

    print("\n  (Los targets con tolerancia los fija investigador-tema en "
          "la ficha de réplica del paper; aqui solo se OBSERVAN.)")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    print(f"Directorio de datos: {DATA_DIR}\n")
    pro = obtener_prostate()
    print()
    com = obtener_communities()
    print("\n== Resumen ==")
    print(f"prostate.csv    (paper)   : {pro.shape[0]} x {pro.shape[1]}")
    print(f"communities.csv (negocio) : {com.shape[0]} x {com.shape[1]}")
    for p in (PROSTATE_FILE, COMM_FILE):
        if os.path.exists(p):
            print(f"  SHA256({os.path.basename(p)}) = {sha256_file(p)}")
    if "--smoke" in argv:
        smoke()
    print("\nVerificacion de fuentes: 18/07/2026. Todo OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
