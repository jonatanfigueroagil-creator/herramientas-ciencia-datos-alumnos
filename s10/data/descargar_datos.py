# -*- coding: utf-8 -*-
"""
Descarga y verificacion de los datos de la Sesion S10
(Clasificadores clasicos: LDA/QDA + Naive Bayes + clases desbalanceadas) del
curso UPC "Herramientas para la Ciencia de Datos".

DOS conjuntos:

  1) IRIS  (REPLICACION LDA; Fisher 1936)  ->  "iris.csv"   150 x 5
     El dataset con el que Fisher introdujo el ANALISIS DISCRIMINANTE LINEAL.
     Cuatro medidas morfologicas (en cm) de 150 flores de iris, 50 de cada una
     de tres especies (setosa, versicolor, virginica).
       Variables: sepal_length_cm, sepal_width_cm, petal_length_cm,
                  petal_width_cm  ->  species {setosa, versicolor, virginica}.
     Fuente PRIMARIA (offline, deterministica): sklearn.datasets.load_iris
     (dataset integrado; NO requiere red). Se materializa como "iris.csv".
     Fuente del silabo / verificacion cruzada: UCI id 53
       https://archive.ics.uci.edu/dataset/53/iris
       (endpoints: .../static/public/53/iris.zip  y
        .../ml/machine-learning-databases/iris/iris.data).
     NOTA (discrepancia conocida de Iris): sklearn.load_iris coincide con la
     version CORREGIDA de Fisher (bezdekIris.data). Difiere de la copia
     historica UCI "iris.data" en EXACTAMENTE 2 filas (dif. maxima 0.5 cm en
     una medida). No altera la frontera LDA/QDA. La verificacion cruzada con
     UCI es NO fatal: si no hay red, se omite.
     Es liviano (~4 KB) -> se conserva el archivo COMPLETO.
     Cita: Fisher, R. A. (1936). "The Use of Multiple Measurements in Taxonomic
     Problems". Annals of Eugenics 7(2), 179-188. Datos: Anderson, E. (1935),
     Gaspe Peninsula. Licencia: dominio publico (dataset clasico).

  2) CREDIT CARD FRAUD (ULB)  (NEGOCIO; desbalance 99:1) -> muestra + completo
     284807 transacciones x 31 columnas; 492 fraudes (Class=1) = 0.1727 %.
     Transacciones de tarjetas de credito europeas de sept. 2013; las 28
     variables V1..V28 son componentes PCA anonimizadas (confidencialidad), mas
     Time (segundos desde la 1.a transaccion), Amount (importe) y Class (0/1).
     Fuente del silabo: Kaggle mlg-ulb/creditcardfraud -> requiere login
     (decision del curso 17/07/2026: sin token de API).
     El COMPLETO pesa ~150.8 MB (>25 MB)  ->  NUNCA se guarda en OneDrive.
       - MIRROR ABIERTO resuelto 19/07/2026 (descarga real validada):
         HuggingFace datasets  David-Egea/Creditcard-fraud-detection
         .../resolve/main/creditcard.csv
         Contenido identico al CSV oficial de Kaggle: 284807 x 31, cols
         Time, V1..V28, Amount, Class; 492 fraudes; 0 nulos.
         SHA256(fuente completa) =
           76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89
       - El completo se descarga SOLO en runtime a un CACHE TEMPORAL del sistema
         (fuera de OneDrive); no se conserva en la carpeta del curso.
       - MUESTRA DE LABORATORIO (<=1 MB, se versiona): TODOS los 492 fraudes +
         una submuestra aleatoria de 1350 transacciones legitimas
         (random_state=42) -> "creditcard_muestra.csv" (1842 filas).
         ATENCION: el ratio de la muestra (~26.7 % fraude) NO es el 99:1 del
         original; es una muestra util para trabajar LOCAL. El desbalance 99:1
         real (0.17 %) se trabaja sobre el COMPLETO en Colab/cache (--full).
     Si el mirror cayera, ver data/README_datos.md (descarga manual Kaggle) y
     dejar "creditcard.csv" en esta carpeta o en el cache: el script lo usa.
     Cita: Dal Pozzolo, A., Caelen, O., Johnson, R. A. & Bontempi, G. (2015).
     "Calibrating Probability with Undersampling for Unbalanced Classification".
     IEEE SSCI. Colaboracion Worldline - MLG, ULB. Licencia: DbCL (uso libre,
     credito a ULB/Worldline).

Idempotente: si un archivo ya existe y su esquema (filas x columnas + columnas
clave + nro de fraudes) es correcto, NO se vuelve a generar. Funciona en local
(Windows/venv) y en Google Colab (DATA_DIR = cwd). Cada archivo se valida y se
reporta su SHA256. El completo de fraude se cachea FUERA de OneDrive.

Uso:
    "<python_del_venv>" descargar_datos.py            # genera + verifica esquema
    "<python_del_venv>" descargar_datos.py --smoke     # + smoke de viabilidad
    "<python_del_venv>" descargar_datos.py --full       # baja el completo al cache
    "<python_del_venv>" descargar_datos.py --verify-uci  # + verificacion cruzada UCI 53
    # o en Colab:  %run descargar_datos.py

Fecha de verificacion de fuentes: 19/07/2026.
"""
import hashlib
import io
import os
import sys
import tempfile
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
HEADERS = {"User-Agent": "Mozilla/5.0 (curso UPC S10 descargar_datos)"}

if "google.colab" in sys.modules:
    DATA_DIR = os.getcwd()
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Cache del CSV completo de fraude FUERA de OneDrive (temporal del sistema).
CACHE_DIR = os.path.join(tempfile.gettempdir(), "curso_upc_s10_creditcard")

# --- 1) IRIS (replicacion LDA, Fisher 1936) ---
IRIS_FILE = os.path.join(DATA_DIR, "iris.csv")
IRIS_COLS = ["sepal_length_cm", "sepal_width_cm",
             "petal_length_cm", "petal_width_cm", "species"]
IRIS_FEATS = IRIS_COLS[:4]
IRIS_SPECIES = ["setosa", "versicolor", "virginica"]
IRIS_N = 150
# Verificacion cruzada UCI 53 (NO fatal si no hay red)
IRIS_UCI_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
)

# --- 2) CREDIT CARD FRAUD (ULB) ---
CC_SAMPLE_FILE = os.path.join(DATA_DIR, "creditcard_muestra.csv")
CC_CACHE_FILE = os.path.join(CACHE_DIR, "creditcard.csv")
# Mirror abierto (HuggingFace); contenido identico al CSV oficial de Kaggle.
CC_URL = (
    "https://huggingface.co/datasets/David-Egea/"
    "Creditcard-fraud-detection/resolve/main/creditcard.csv"
)
CC_SHA256_SRC = (
    "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
)
CC_COLS = (["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"])
CC_N = 284807
CC_NCOLS = 31
CC_FRAUDES = 492
# Muestra de laboratorio: TODOS los fraudes + N legitimas (rs=42) -> <=1 MB
CC_N_LEGIT = 1350
CC_RANDOM_STATE = 42
CC_SAMPLE_N = CC_FRAUDES + CC_N_LEGIT   # 1842


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str) -> bytes:
    if requests is None:
        raise RuntimeError("requests no disponible")
    r = requests.get(url, headers=HEADERS, timeout=600)
    r.raise_for_status()
    return r.content


# ---------------------------------------------------------------------------
# 1) IRIS (replicacion LDA, Fisher 1936)
# ---------------------------------------------------------------------------
def obtener_iris(verify_uci: bool = False) -> pd.DataFrame:
    """Genera (si hace falta) iris.csv desde sklearn.load_iris (offline) y
    devuelve el DataFrame verificado (150 x 5). Verificacion cruzada UCI 53
    opcional y NO fatal."""
    print("== IRIS 150x5 (REPLICACION LDA; Fisher 1936; sklearn built-in) ==")
    if _iris_valido():
        print(f"  = presente y verificado: {IRIS_FILE}")
        df = pd.read_csv(IRIS_FILE)
    else:
        from sklearn.datasets import load_iris
        d = load_iris(as_frame=True)
        df = d.data.copy()
        df.columns = IRIS_FEATS
        df["species"] = pd.Categorical.from_codes(d.target, IRIS_SPECIES)
        if df.shape != (IRIS_N, 5):
            raise ValueError(f"IRIS: esquema {df.shape} != ({IRIS_N}, 5)")
        df.to_csv(IRIS_FILE, index=False, encoding="utf-8")
        print(f"  + generado desde sklearn.load_iris -> {IRIS_FILE}")

    vc = df["species"].value_counts().to_dict()
    if len(df) != IRIS_N or set(vc) != set(IRIS_SPECIES) or set(vc.values()) != {50}:
        raise ValueError(f"IRIS: distribucion inesperada {vc}")
    print(f"  = verificado: {len(df)} filas x {df.shape[1]} cols; "
          f"especies {vc}")

    if verify_uci:
        _verificar_uci_iris(df)
    return df


def _iris_valido() -> bool:
    if not os.path.exists(IRIS_FILE):
        return False
    try:
        df = pd.read_csv(IRIS_FILE, nrows=5)
        return list(df.columns) == IRIS_COLS
    except Exception:  # noqa: BLE001
        return False


def _verificar_uci_iris(df_sklearn: pd.DataFrame) -> None:
    """Verificacion cruzada NO fatal contra UCI 53 (iris.data). Documenta la
    discrepancia conocida de 2 filas frente a la version corregida de sklearn."""
    try:
        raw = _fetch(IRIS_UCI_DATA_URL)
        uci = pd.read_csv(io.BytesIO(raw), header=None).dropna()
        a = uci.iloc[:, :4].to_numpy(dtype=float)
        b = df_sklearn[IRIS_FEATS].to_numpy(dtype=float)
        if a.shape != b.shape:
            print(f"  ~ UCI 53: forma {a.shape} != {b.shape} (se omite cruce)")
            return
        difs = int((np.abs(a - b).sum(axis=1) > 0).sum())
        print(f"  = verificacion cruzada UCI 53: {len(uci)} filas; "
              f"filas que difieren de sklearn = {difs} "
              f"(esperado 2 = discrepancia historica conocida)")
    except Exception as e:  # noqa: BLE001
        print(f"  ~ verificacion cruzada UCI 53 omitida (sin red / {e})")


# ---------------------------------------------------------------------------
# 2) CREDIT CARD FRAUD - completo NUNCA en OneDrive
# ---------------------------------------------------------------------------
def _obtener_completo_fraude() -> str:
    """Descarga (o reutiliza el cache temporal) el creditcard.csv completo.
    Devuelve la ruta al CSV en el cache (fuera de OneDrive). Idempotente por
    SHA256. Si existe un creditcard.csv provisto manualmente (descarga Kaggle)
    en DATA_DIR o en CACHE_DIR, se usa ese."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    # (a) archivo manual del docente en la carpeta de la sesion (Kaggle)
    manual = os.path.join(DATA_DIR, "creditcard.csv")
    if os.path.exists(manual):
        print(f"  = usando creditcard.csv provisto en la carpeta ({manual})")
        return manual
    # (b) cache temporal ya descargado
    if os.path.exists(CC_CACHE_FILE) and sha256_file(CC_CACHE_FILE) == CC_SHA256_SRC:
        print(f"  = completo en cache temporal ({CC_CACHE_FILE})")
        return CC_CACHE_FILE
    # (c) descargar del mirror abierto
    print(f"  ... descargando completo (~150.8 MB, SOLO runtime; Kaggle pide login)\n"
          f"      {CC_URL}")
    raw = _fetch(CC_URL)
    sha = sha256_bytes(raw)
    aviso = "" if sha == CC_SHA256_SRC else "  [SHA256 de fuente NUEVO -> revisar]"
    with open(CC_CACHE_FILE, "wb") as f:
        f.write(raw)
    print(f"  ... completo -> {CC_CACHE_FILE}  ({len(raw)/1e6:.1f} MB)  "
          f"SHA256={sha[:16]}...{aviso}")
    return CC_CACHE_FILE


def cargar_fraude_completo() -> pd.DataFrame:
    """Devuelve el DataFrame COMPLETO de fraude (284807 x 31). No lo guarda en
    OneDrive. Usar para trabajar el desbalance real 99:1 en Colab/cache."""
    path = _obtener_completo_fraude()
    df = pd.read_csv(path)
    faltan = [c for c in ["Time", "V1", "Amount", "Class"] if c not in df.columns]
    if faltan:
        raise ValueError(f"fraude completo: faltan columnas {faltan}")
    if df.shape != (CC_N, CC_NCOLS):
        raise ValueError(f"fraude completo: {df.shape} != ({CC_N}, {CC_NCOLS})")
    nf = int((df["Class"] == 1).sum())
    if nf != CC_FRAUDES:
        raise ValueError(f"fraude completo: {nf} fraudes != {CC_FRAUDES}")
    print(f"  = completo verificado: {df.shape[0]} x {df.shape[1]}; "
          f"fraudes={nf} ({df['Class'].mean()*100:.4f}%); "
          f"nulos={int(df.isna().sum().sum())}")
    return df


def obtener_fraude_muestra() -> pd.DataFrame:
    """Genera (si hace falta) la muestra de laboratorio <=1 MB: TODOS los 492
    fraudes + 1350 legitimas (random_state=42), ordenada por Time."""
    print("== CREDIT CARD FRAUD - MUESTRA de laboratorio (ULB; completo NUNCA en OneDrive) ==")
    if _muestra_valida():
        df = pd.read_csv(CC_SAMPLE_FILE)
        print(f"  = presente y verificada: {CC_SAMPLE_FILE} "
              f"({len(df)} filas, fraudes={int((df['Class']==1).sum())})")
        return df

    full = cargar_fraude_completo()
    fraude = full[full["Class"] == 1]
    legit = full[full["Class"] == 0].sample(n=CC_N_LEGIT, random_state=CC_RANDOM_STATE)
    muestra = (pd.concat([fraude, legit])
               .sort_values("Time")
               .reset_index(drop=True))
    muestra.to_csv(CC_SAMPLE_FILE, index=False, encoding="utf-8")
    mb = os.path.getsize(CC_SAMPLE_FILE) / 1e6
    print(f"  + muestra {CC_FRAUDES} fraudes + {CC_N_LEGIT} legitimas "
          f"(rs={CC_RANDOM_STATE}) -> {CC_SAMPLE_FILE} "
          f"({len(muestra)} filas, {mb:.3f} MB, fraude={muestra['Class'].mean()*100:.2f}%)")
    if mb > 1.0:
        print(f"  ! AVISO: la muestra pesa {mb:.3f} MB (>1 MB). Reducir CC_N_LEGIT.")
    return muestra


def _muestra_valida() -> bool:
    if not os.path.exists(CC_SAMPLE_FILE):
        return False
    try:
        df = pd.read_csv(CC_SAMPLE_FILE)
        if list(df.columns) != CC_COLS:
            return False
        if len(df) != CC_SAMPLE_N:
            return False
        if int((df["Class"] == 1).sum()) != CC_FRAUDES:
            return False
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# SMOKE de VIABILIDAD (reporta valores OBSERVADOS; NO define targets)
# Los targets con tolerancia los fija investigador-tema en
# la ficha de réplica del paper (dueno unico).
# ---------------------------------------------------------------------------
def smoke():
    from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 recall_score, f1_score)
    from imblearn.over_sampling import SMOTE

    print("\n########## SMOKE DE VIABILIDAD S10 ##########")

    # -- (a) IRIS: LDA y QDA con validacion cruzada estratificada 5-fold --
    print("\n== (a) IRIS: LDA vs QDA (StratifiedKFold 5, rs=42) accuracy ==")
    iris = obtener_iris()
    Xi = iris[IRIS_FEATS].values
    yi = iris["species"].astype("category").cat.codes.values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_lda = cross_val_score(LinearDiscriminantAnalysis(), Xi, yi, cv=cv).mean()
    acc_qda = cross_val_score(QuadraticDiscriminantAnalysis(), Xi, yi, cv=cv).mean()
    print(f"   LDA accuracy OBSERVADO = {acc_lda:.4f}")
    print(f"   QDA accuracy OBSERVADO = {acc_qda:.4f}")

    # -- (b) FRAUDE (completo 99:1): paradoja de la accuracy + PR-AUC + SMOTE --
    print("\n== (b) FRAUDE completo 99:1: trivial recall, PR-AUC base, efecto SMOTE ==")
    df = cargar_fraude_completo()
    X = df.drop(columns="Class").values
    y = df["Class"].values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    triv_acc = (yte == 0).mean()
    triv_rec = recall_score(yte, np.zeros_like(yte))
    print(f"   TRIVIAL 'todo legitimo': accuracy={triv_acc:.4f}  "
          f"recall(fraude) OBSERVADO = {triv_rec:.4f}  (paradoja de la accuracy)")

    for name, clf in [
        ("LogReg", LogisticRegression(max_iter=1000)),
        ("LDA", LinearDiscriminantAnalysis()),
        ("QDA", QuadraticDiscriminantAnalysis(reg_param=0.01)),
        ("GaussianNB", GaussianNB()),
    ]:
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        yp = (p >= 0.5).astype(int)
        print(f"   {name:11s} PR-AUC={average_precision_score(yte, p):.4f}  "
              f"ROC-AUC={roc_auc_score(yte, p):.4f}  "
              f"recall={recall_score(yte, yp):.4f}  F1={f1_score(yte, yp):.4f}")

    base = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    rec_base = recall_score(yte, (base.predict_proba(Xte)[:, 1] >= 0.5).astype(int))
    Xr, yr = SMOTE(random_state=42).fit_resample(Xtr, ytr)
    smo = LogisticRegression(max_iter=1000).fit(Xr, yr)
    rec_smote = recall_score(yte, (smo.predict_proba(Xte)[:, 1] >= 0.5).astype(int))
    print(f"   SMOTE: recall(fraude) LogReg base={rec_base:.4f} -> "
          f"post-SMOTE={rec_smote:.4f}  (sube el recall de la minoritaria)")

    print("\n  (Los targets con tolerancia los fija investigador-tema en "
          "la ficha de réplica del paper; aqui solo se OBSERVAN.)")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    print(f"Directorio de datos (OneDrive): {DATA_DIR}")
    print(f"Cache del completo (temporal):  {CACHE_DIR}\n")

    iris = obtener_iris(verify_uci="--verify-uci" in argv)
    print()
    muestra = obtener_fraude_muestra()

    print("\n== Resumen ==")
    print(f"iris.csv               : {iris.shape[0]} x {iris.shape[1]}  "
          f"(replicacion LDA; Fisher 1936)")
    print(f"creditcard_muestra.csv : {muestra.shape[0]} x {muestra.shape[1]}  "
          f"(negocio; fraude={muestra['Class'].mean()*100:.2f}% en la MUESTRA; "
          f"completo 284807x31 = 0.17% NO en OneDrive)")

    print("\n== Checksums SHA256 (archivo local) ==")
    for p in (IRIS_FILE, CC_SAMPLE_FILE):
        if os.path.exists(p):
            mb = os.path.getsize(p) / 1e6
            print(f"  {os.path.basename(p):24s} {sha256_file(p)}  ({mb:.3f} MB)")
    print(f"  {'creditcard.csv (fuente)':24s} {CC_SHA256_SRC}  (completo ~150.8 MB, NO en OneDrive)")

    if "--full" in argv:
        print()
        cargar_fraude_completo()
    if "--smoke" in argv:
        smoke()
    print("\nVerificacion de fuentes: 19/07/2026. Todo OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
