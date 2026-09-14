# -*- coding: utf-8 -*-
"""
Descarga y verificacion de los datos de la Sesion S09
(Regresion Logistica + Modelos Lineales Generalizados / GLM) del curso UPC
"Herramientas para la Ciencia de Datos".

TRES conjuntos (los tres pesan < 25 MB -> se CONSERVAN completos en la carpeta;
ninguno necesita muestra):

  1) SAheart  (REPLICACION logistica; ESL cap. 4 / Hastie, Tibshirani & Friedman)
     -> "SAheart.data"   462 filas x 10 variables (+ chd).
     South African Heart Disease: estudio retrospectivo de cardiopatia coronaria
     (CHD) en una region de alto riesgo del Cabo Occidental (Rousseauw et al.,
     1983). Es el dataset con el que "The Elements of Statistical Learning"
     ilustra la regresion logistica (Tabla 4.2: coeficientes en log-odds de chd
     sobre los factores de riesgo).
       Variables: sbp, tobacco, ldl, adiposity, famhist, typea, obesity,
                  alcohol, age  ->  chd (0/1).
       famhist es categorica {Present, Absent}; el resto numericas.
     Fuente del silabo: https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data
       -> HTTP-403 a scripts (bloqueo anti-bot; 18/07/2026).
     FALLBACK aplicado: mirror abierto en GitHub del mismo archivo (contenido
     identico, con cabecera y columna row.names):
       empathy87/The-Elements-of-Statistical-Learning-Python-Notebooks,
       data/South African Heart Disease.txt
     Se guarda LOCAL como "SAheart.data" en el formato canonico de Hastie
     (CSV con row.names). Verificado 18/07/2026: 462 filas, esquema completo.
     Cita: Hastie, T., Tibshirani, R. & Friedman, J. (2009). The Elements of
     Statistical Learning, 2.a ed., cap. 4.4. Springer. Datos: Rousseauw, J.
     et al. (1983). South African Medical Journal 64, 430-436.
     Licencia: dominio publico academico (dataset de libro de texto).

  2) ships  (REPLICACION GLM Poisson; McCullagh & Nelder 1989 / paquete R MASS)
     -> "ships.csv"   40 filas x 5 variables.
     Averias (incidents) en cascos de barcos de carga segun tipo, ano de
     construccion y periodo de operacion, con la EXPOSICION (service = meses-
     buque de servicio agregados). Es el ejemplo canonico de regresion de
     Poisson con offset log(exposure) de McCullagh & Nelder.
       Variables: type {A..E}, year, period, service, incidents.
     Fuente: Rdatasets (mirror de datasets de R), paquete MASS:
       https://vincentarelbundock.github.io/Rdatasets/csv/MASS/ships.csv
     Verificado 18/07/2026: 40 filas, esquema completo.
     Cita: McCullagh, P. & Nelder, J. A. (1989). Generalized Linear Models,
     2.a ed. Chapman & Hall. Paper de la familia GLM: Nelder & Wedderburn
     (1972), JRSS-A 135(3). Licencia: MASS (GPL-2/3), uso academico.

  3) Telco Customer Churn  (NEGOCIO / churn) -> "telco_churn.csv"
     7043 filas x 21 columnas. Datos de muestra de IBM (Watson Analytics):
     cada fila es un cliente de una telco; target Churn {Yes, No}. Se usa para
     el laboratorio de negocio (modelo de fuga con logistica + ROC/AUC).
     Fuente del silabo: Kaggle blastchar/telco-customer-churn -> requiere login
     (decision del curso 17/07/2026: sin token de API).
     FALLBACK aplicado: mirror abierto en GitHub del mismo CSV oficial
     (WA_Fn-UseC_-Telco-Customer-Churn.csv):
       treselle-systems/customer_churn_analysis
     LIMPIEZA DOCUMENTADA de TotalCharges: la columna llega como texto; 11 filas
     (clientes con tenure = 0, altas del mes) traen un ESPACIO en blanco en vez
     de un numero. Al convertir a numerico esos 11 espacios pasan a NaN. El
     notebook los elimina (o imputa 0) -> el modelo queda con 7032 filas.
     Verificado 18/07/2026: 7043 filas, 21 columnas, Churn = 1869 Yes / 5174 No.
     Cita: IBM (2019). Telco Customer Churn (IBM Cognos / Watson Analytics
     sample). Licencia: muestra publica de IBM para fines educativos.

Idempotente: si un archivo ya existe y su esquema es correcto, NO se vuelve a
descargar. Funciona en local (Windows/venv) y en Google Colab (DATA_DIR = cwd).
Cada archivo se valida por filas x columnas y se reporta su SHA256.

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
import warnings

warnings.filterwarnings("ignore")

try:
    import requests
except ImportError:  # en Colab requests viene preinstalado
    requests = None

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (curso UPC S09 descargar_datos)"}

if "google.colab" in sys.modules:
    DATA_DIR = os.getcwd()
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1) SAheart (replicacion logistica, ESL cap. 4) ---
SAHEART_FILE = os.path.join(DATA_DIR, "SAheart.data")
SAHEART_URL = (
    "https://raw.githubusercontent.com/empathy87/"
    "The-Elements-of-Statistical-Learning-Python-Notebooks/master/"
    "data/South%20African%20Heart%20Disease.txt"
)
SAHEART_SHA256_SRC = (
    "c158781284d05a24265bb3764e0c48ad9e0dc9f646b27b0b44d43eba46299805"
)
SAHEART_N = 462
SAHEART_COLS = ["sbp", "tobacco", "ldl", "adiposity", "famhist", "typea",
                "obesity", "alcohol", "age", "chd"]

# --- 2) ships (replicacion GLM Poisson, McCullagh & Nelder 1989) ---
SHIPS_FILE = os.path.join(DATA_DIR, "ships.csv")
SHIPS_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/ships.csv"
SHIPS_SHA256_SRC = (
    "447f68b31465fb092dc542bdc7a0ddc0b80f5b983314b29b0667b25a11b5f115"
)
SHIPS_N = 40
SHIPS_COLS = ["type", "year", "period", "service", "incidents"]

# --- 3) Telco Customer Churn (negocio) ---
TELCO_FILE = os.path.join(DATA_DIR, "telco_churn.csv")
TELCO_URL = (
    "https://raw.githubusercontent.com/treselle-systems/"
    "customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
TELCO_SHA256_SRC = (
    "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
)
TELCO_N = 7043
TELCO_NCOLS = 21
TELCO_KEY_COLS = ["customerID", "tenure", "MonthlyCharges", "TotalCharges",
                  "Contract", "Churn"]


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
    r = requests.get(url, headers=HEADERS, timeout=300)
    r.raise_for_status()
    return r.content


def _guardar(path: str, raw: bytes, sha_esperado: str) -> None:
    """Guarda bytes crudos en UTF-8 con saltos \n; avisa si el SHA de la fuente
    cambio respecto al verificado el 18/07/2026."""
    sha = sha256_bytes(raw)
    aviso = "" if sha == sha_esperado else "  [SHA256 de fuente NUEVO -> revisar]"
    txt = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    print(f"  + guardado -> {os.path.basename(path)}  ({len(raw)/1e6:.3f} MB)"
          f"  SHA256(fuente)={sha[:16]}...{aviso}")


# ---------------------------------------------------------------------------
# 1) SAheart
# ---------------------------------------------------------------------------
def obtener_saheart() -> pd.DataFrame:
    """Descarga (si hace falta) SAheart.data y devuelve el DataFrame verificado."""
    print("== SAheart 462 filas (REPLICACION logistica; ESL cap. 4, Tabla 4.2) ==")
    if not os.path.exists(SAHEART_FILE):
        print(f"  ... descargando mirror GitHub (hastie.su.domains da 403)\n"
              f"      {SAHEART_URL}")
        _guardar(SAHEART_FILE, _fetch(SAHEART_URL), SAHEART_SHA256_SRC)

    df = pd.read_csv(SAHEART_FILE)
    if "row.names" in df.columns:
        df = df.drop(columns=["row.names"])
    faltan = set(SAHEART_COLS) - set(df.columns)
    if faltan:
        raise ValueError(f"SAheart: faltan columnas {faltan}")
    if len(df) != SAHEART_N:
        raise ValueError(f"SAheart: {len(df)} filas != {SAHEART_N}")
    if set(df["famhist"].unique()) != {"Present", "Absent"}:
        raise ValueError(f"SAheart: famhist inesperado {df['famhist'].unique()}")
    if set(df["chd"].unique()) != {0, 1}:
        raise ValueError(f"SAheart: chd no es binario {df['chd'].unique()}")
    print(f"  = verificado: {len(df)} filas x {df.shape[1]} vars; "
          f"chd={int(df['chd'].sum())}/{len(df)} casos; "
          f"famhist Present={(df['famhist']=='Present').sum()}")
    return df


# ---------------------------------------------------------------------------
# 2) ships
# ---------------------------------------------------------------------------
def obtener_ships() -> pd.DataFrame:
    """Descarga (si hace falta) ships.csv y devuelve el DataFrame verificado."""
    print("== ships 40 filas (REPLICACION GLM Poisson; McCullagh & Nelder 1989) ==")
    if not os.path.exists(SHIPS_FILE):
        print(f"  ... descargando Rdatasets (paquete MASS)\n      {SHIPS_URL}")
        _guardar(SHIPS_FILE, _fetch(SHIPS_URL), SHIPS_SHA256_SRC)

    df = pd.read_csv(SHIPS_FILE)
    # Rdatasets antepone una columna de indice (rownames); se ignora.
    faltan = set(SHIPS_COLS) - set(df.columns)
    if faltan:
        raise ValueError(f"ships: faltan columnas {faltan}")
    if len(df) != SHIPS_N:
        raise ValueError(f"ships: {len(df)} filas != {SHIPS_N}")
    n_exp = int((df["service"] > 0).sum())
    print(f"  = verificado: {len(df)} filas; tipos={sorted(df['type'].unique())}; "
          f"{n_exp} filas con exposicion (service>0); "
          f"total incidents={int(df['incidents'].sum())}")
    return df


# ---------------------------------------------------------------------------
# 3) Telco Customer Churn
# ---------------------------------------------------------------------------
def obtener_telco() -> pd.DataFrame:
    """Descarga (si hace falta) telco_churn.csv y devuelve el DataFrame verificado
    (CRUDO, sin limpiar). La limpieza de TotalCharges la hace limpiar_telco()."""
    print("== Telco Customer Churn 7043 filas (NEGOCIO; IBM sample; mirror abierto) ==")
    if not os.path.exists(TELCO_FILE):
        print(f"  ... descargando mirror GitHub (Kaggle requiere login)\n"
              f"      {TELCO_URL}")
        _guardar(TELCO_FILE, _fetch(TELCO_URL), TELCO_SHA256_SRC)

    df = pd.read_csv(TELCO_FILE)
    faltan = set(TELCO_KEY_COLS) - set(df.columns)
    if faltan:
        raise ValueError(f"Telco: faltan columnas {faltan}")
    if len(df) != TELCO_N or df.shape[1] != TELCO_NCOLS:
        raise ValueError(f"Telco: {df.shape} != ({TELCO_N}, {TELCO_NCOLS})")
    if set(df["Churn"].unique()) != {"Yes", "No"}:
        raise ValueError(f"Telco: Churn inesperado {df['Churn'].unique()}")
    n_espacios = (df["TotalCharges"].astype(str).str.strip() == "").sum()
    vc = df["Churn"].value_counts().to_dict()
    print(f"  = verificado: {len(df)} filas x {df.shape[1]} cols; "
          f"Churn={vc}; TotalCharges con espacio en blanco (tenure=0)={n_espacios}")
    return df


def limpiar_telco(df: pd.DataFrame, imputar_cero: bool = False) -> pd.DataFrame:
    """Limpieza documentada de TotalCharges: texto -> numerico; los 11 espacios
    en blanco (clientes con tenure=0) pasan a NaN. Por defecto se ELIMINAN
    (imputar_cero=False -> 7032 filas); imputar_cero=True los pone a 0."""
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(
        out["TotalCharges"].astype(str).str.strip().replace("", np.nan)
    )
    n_na = int(out["TotalCharges"].isna().sum())
    if imputar_cero:
        out["TotalCharges"] = out["TotalCharges"].fillna(0.0)
        print(f"  limpieza Telco: {n_na} NaN imputados a 0 -> {len(out)} filas")
    else:
        out = out.dropna(subset=["TotalCharges"]).reset_index(drop=True)
        print(f"  limpieza Telco: {n_na} NaN eliminados -> {len(out)} filas")
    return out


# ---------------------------------------------------------------------------
# SMOKE de VIABILIDAD (reporta valores OBSERVADOS; NO define targets)
# Los targets con tolerancia los fija investigador-tema en
# la ficha de réplica del paper (dueno unico).
# ---------------------------------------------------------------------------
def smoke():
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, confusion_matrix

    print("\n########## SMOKE DE VIABILIDAD S09 (statsmodels + sklearn) ##########")

    # -- (a) SAheart: Logit (chd ~ todos), famhist a dummy --
    print("\n== (a) statsmodels Logit: chd ~ todos (SAheart, famhist dummy) ==")
    sa = obtener_saheart()
    sa["famhist"] = (sa["famhist"] == "Present").astype(int)
    X = sm.add_constant(sa.drop(columns=["chd"]))
    modelo = sm.Logit(sa["chd"], X).fit(disp=0)
    print("   coeficientes OBSERVADOS (log-odds) y odds ratio:")
    for k in ["tobacco", "famhist", "age", "ldl"]:
        print(f"     {k:8s} coef={modelo.params[k]:+.5f}  "
              f"OR={np.exp(modelo.params[k]):.4f}  p={modelo.pvalues[k]:.4f}")
    print(f"   pseudo-R2 McFadden = {modelo.prsquared:.4f}")

    # -- (b) ships: GLM Poisson con offset log(service) --
    print("\n== (b) statsmodels GLM(Poisson): incidents ~ type+year+period, "
          "offset=log(service) (ships) ==")
    sh = obtener_ships()
    sh = sh[sh["service"] > 0].copy()
    glm = smf.glm("incidents ~ C(type) + C(year) + C(period)", data=sh,
                  family=sm.families.Poisson(),
                  offset=np.log(sh["service"])).fit()
    ratio = glm.deviance / glm.df_resid
    print(f"   deviance={glm.deviance:.2f}  df_resid={glm.df_resid}  "
          f"deviance/df={ratio:.3f}  (>1 sugiere sobre-dispersion)")
    print("   coeficientes OBSERVADOS (log-tasa):")
    for k in ["Intercept", "C(type)[T.B]", "C(type)[T.C]",
              "C(type)[T.E]", "C(period)[T.75]"]:
        if k in glm.params:
            print(f"     {k:16s} coef={glm.params[k]:+.4f}  "
                  f"tasa x{np.exp(glm.params[k]):.3f}")

    # -- (c) Telco: LogisticRegression sklearn -> AUC --
    print("\n== (c) sklearn LogisticRegression: churn (Telco, one-hot, "
          "train/test 75/25) -> AUC ==")
    tc = limpiar_telco(obtener_telco())
    y = (tc["Churn"] == "Yes").astype(int)
    tc = tc.drop(columns=["Churn", "customerID"])
    num = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    cat = [c for c in tc.columns if c not in num]
    Xd = pd.get_dummies(tc, columns=cat, drop_first=True)
    Xtr, Xte, ytr, yte = train_test_split(
        Xd, y, test_size=0.2, random_state=42, stratify=y)  # 0,2 = split operativo (notebook/validador)
    sc = StandardScaler().fit(Xtr[num])
    Xtr, Xte = Xtr.copy(), Xte.copy()
    Xtr[num] = sc.transform(Xtr[num])
    Xte[num] = sc.transform(Xte[num])
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)
    cm = confusion_matrix(yte, (proba >= 0.5).astype(int))
    print(f"   n modelo={len(tc)}  features one-hot={Xd.shape[1]}  "
          f"churn_rate={y.mean():.3f}")
    print(f"   AUC(test) OBSERVADO = {auc:.4f}")
    print(f"   matriz de confusion (umbral 0.5) [[TN,FP],[FN,TP]] = "
          f"{cm.tolist()}")

    print("\n  (Los targets con tolerancia los fija investigador-tema en "
          "la ficha de réplica del paper; aqui solo se OBSERVAN.)")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    print(f"Directorio de datos: {DATA_DIR}\n")

    sa = obtener_saheart()
    print()
    sh = obtener_ships()
    print()
    tc = obtener_telco()

    print("\n== Resumen ==")
    print(f"SAheart.data     : {len(sa)} filas x {sa.shape[1]} vars   (replicacion logistica, ESL)")
    print(f"ships.csv        : {len(sh)} filas                    (replicacion GLM Poisson)")
    print(f"telco_churn.csv  : {len(tc)} filas x {tc.shape[1]} cols  (negocio / churn)")

    print("\n== Checksums SHA256 (archivo local) ==")
    for p in (SAHEART_FILE, SHIPS_FILE, TELCO_FILE):
        if os.path.exists(p):
            mb = os.path.getsize(p) / 1e6
            print(f"  {os.path.basename(p):16s} {sha256_file(p)}  ({mb:.3f} MB)")

    if "--smoke" in argv:
        smoke()
    print("\nVerificacion de fuentes: 18/07/2026. Todo OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
