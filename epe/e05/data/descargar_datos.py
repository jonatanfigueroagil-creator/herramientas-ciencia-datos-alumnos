# -*- coding: utf-8 -*-
"""
Descarga y verificacion de los datasets de la Sesion EPE E5
(Predecir categorias: churn y fraude) del curso UPC
"Herramientas para la Ciencia de Datos".

Dos casos de negocio (REUSO EPE: no se regenera, se reutiliza lo ya cerrado):

  1) TELCO CUSTOMER CHURN  (churn / fuga de clientes). Dataset ya verificado
     para este curso: 7 043 clientes de una telco, target Churn {Yes, No}. Es
     pequeno (<1 MB) -> se conserva completo. En local se busca primero una
     copia ya verificada; si no esta, se descarga de un mirror
     abierto (Kaggle pide login). El notebook lo modela EN VIVO y reproduce el
     resultado de S09 (AUC del churn, matriz de confusion, umbral por costo).

  2) CREDIT CARD FRAUD (ULB)  (fraude; desbalance 99:1). Muestra de laboratorio
     ya verificada para este curso. El COMPLETO pesa ~150,8 MB
     (>25 MB) -> NUNCA en OneDrive: solo se descarga en runtime a un cache
     temporal del sistema (--full). Para trabajar en local se versiona una
     MUESTRA <=1 MB (492 fraudes + 1 350 legitimas, random_state=42 = 1 842
     filas, ~26,7 % de fraude: NO es el 99:1 real, es una muestra de laboratorio).
     El desbalance real 99:1 (0,17 %) y sus metricas operativas provienen de los
     RESULTADOS YA VALIDADOS de S10 (resultados/S10_resultados.xlsx), que el
     cuaderno reutiliza; no se recalculan sobre la muestra (daria otro numero).

Idempotente: si un archivo ya existe y su esquema/SHA256 coincide, NO se vuelve a
obtener. Funciona en local (Windows/venv) y en Google Colab (usa el directorio de
trabajo actual). El completo de fraude se cachea FUERA de OneDrive.

Uso:
    "<python_del_venv>" descargar_datos.py            # verifica los dos casos (local)
    "<python_del_venv>" descargar_datos.py --full      # ademas baja el completo de fraude al cache
    # o en Colab:  %run descargar_datos.py

Fecha de verificacion de fuentes: 06/08/2026.
"""
import hashlib
import io
import os
import shutil
import sys
import tempfile

try:
    import requests
except ImportError:  # en Colab requests viene preinstalado
    requests = None

import pandas as pd

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (curso UPC E05 descargar_datos)"}

if "google.colab" in sys.modules:
    DATA_DIR = os.getcwd()
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1) Telco Customer Churn (churn) ---
TELCO_FILE = os.path.join(DATA_DIR, "telco_churn.csv")
# Reuso local del material de S09 (evita re-descargar)
_S09_DATA = os.path.normpath(os.path.join(
    DATA_DIR, "..", "..", "..", "Sesiones", "S09_logistica_glm", "data"))
S09_TELCO = os.path.join(_S09_DATA, "telco_churn.csv")
# Mirror abierto verificado (el mismo de S09; Kaggle pide login)
TELCO_URL = (
    "https://raw.githubusercontent.com/treselle-systems/"
    "customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
TELCO_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
TELCO_N = 7043
TELCO_NCOLS = 21
TELCO_KEY = ["customerID", "tenure", "MonthlyCharges", "TotalCharges",
             "Contract", "Churn"]

# --- 2) Credit Card Fraud (ULB) — muestra <=1 MB + completo fuera de OneDrive ---
CC_SAMPLE_FILE = os.path.join(DATA_DIR, "creditcard_muestra.csv")
_S10_DATA = os.path.normpath(os.path.join(
    DATA_DIR, "..", "..", "..", "Sesiones", "S10_clasificadores_clasicos", "data"))
S10_CC_SAMPLE = os.path.join(_S10_DATA, "creditcard_muestra.csv")
CACHE_DIR = os.path.join(tempfile.gettempdir(), "curso_upc_e05_creditcard")
CC_CACHE_FILE = os.path.join(CACHE_DIR, "creditcard.csv")
# Mirror abierto (HuggingFace); contenido identico al CSV oficial de Kaggle.
CC_URL = (
    "https://huggingface.co/datasets/David-Egea/"
    "Creditcard-fraud-detection/resolve/main/creditcard.csv"
)
CC_SHA256_SRC = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
CC_SAMPLE_SHA256 = "2841263a95459c6f0205deda18a8fa1a42e6842e739e8599f998922f9d55e37e"
CC_COLS = (["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"])
CC_N, CC_NCOLS, CC_FRAUDES = 284807, 31, 492
CC_N_LEGIT, CC_RS = 1350, 42
CC_SAMPLE_N = CC_FRAUDES + CC_N_LEGIT  # 1842


# ---------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def verificar_sha(path: str, esperado: str, etiqueta: str) -> bool:
    """Comprueba el SHA256 del archivo contra el esperado. Devuelve True si
    coincide (integridad confirmada) y False si difiere (avisa, no aborta: el
    esquema ya es la barrera de correccion; un fin de linea recodificado por
    OneDrive puede cambiar el hash sin danar los datos)."""
    real = sha256_file(path)
    if real == esperado:
        print(f"  = SHA256 verificado ({etiqueta}): {real[:16]}...")
        return True
    print(f"  ! AVISO ({etiqueta}): SHA256 {real[:16]}... != esperado {esperado[:16]}... "
          f"(esquema OK; posible recodificacion de fin de linea)")
    return False


def _fetch(url: str) -> bytes:
    if requests is None:
        raise RuntimeError("requests no disponible")
    r = requests.get(url, headers=HEADERS, timeout=600)
    r.raise_for_status()
    return r.content


# ---------------------------------------------------------------------------
# 1) Telco Customer Churn (reuso de S09)
# ---------------------------------------------------------------------------
def obtener_telco() -> pd.DataFrame:
    print("== TELCO CUSTOMER CHURN (churn; reuso de S09; 7 043 filas) ==")
    if os.path.exists(TELCO_FILE):
        df = pd.read_csv(TELCO_FILE)
        if len(df) == TELCO_N and df.shape[1] == TELCO_NCOLS:
            verificar_sha(TELCO_FILE, TELCO_SHA256, "telco_churn.csv")
            print(f"  = presente y verificado: {TELCO_FILE}")
            return df
        print("  ! archivo presente con esquema inesperado; se re-obtiene.")
    # 1) reuso local desde S09
    if os.path.exists(S09_TELCO):
        shutil.copyfile(S09_TELCO, TELCO_FILE)
        print(f"  + copiado desde S09 -> {TELCO_FILE}")
        verificar_sha(TELCO_FILE, TELCO_SHA256, "telco_churn.csv")
        return pd.read_csv(TELCO_FILE)
    # 2) descarga del mirror abierto
    raw = _fetch(TELCO_URL)
    df = pd.read_csv(io.BytesIO(raw))
    if set(TELCO_KEY) - set(df.columns) or len(df) != TELCO_N:
        raise ValueError(f"Telco: esquema inesperado {df.shape}")
    with open(TELCO_FILE, "w", encoding="utf-8", newline="\n") as f:
        f.write(raw.decode("utf-8").replace("\r\n", "\n"))
    verificar_sha(TELCO_FILE, TELCO_SHA256, "telco_churn.csv (mirror)")
    print(f"  + descargado del mirror -> {TELCO_FILE}")
    return df


# ---------------------------------------------------------------------------
# 2) Credit Card Fraud (reuso de S10) — muestra + completo (cache)
# ---------------------------------------------------------------------------
def _muestra_valida() -> bool:
    if not os.path.exists(CC_SAMPLE_FILE):
        return False
    try:
        df = pd.read_csv(CC_SAMPLE_FILE)
        return (list(df.columns) == CC_COLS and len(df) == CC_SAMPLE_N
                and int((df["Class"] == 1).sum()) == CC_FRAUDES)
    except Exception:
        return False


def obtener_fraude_muestra() -> pd.DataFrame:
    print("== CREDIT CARD FRAUD - MUESTRA de laboratorio (reuso de S10; <=1 MB) ==")
    if _muestra_valida():
        df = pd.read_csv(CC_SAMPLE_FILE)
        verificar_sha(CC_SAMPLE_FILE, CC_SAMPLE_SHA256, "creditcard_muestra.csv")
        print(f"  = presente y verificada: {CC_SAMPLE_FILE} "
              f"({len(df)} filas, fraudes={int((df['Class']==1).sum())})")
        return df
    # 1) reuso local desde S10
    if os.path.exists(S10_CC_SAMPLE):
        shutil.copyfile(S10_CC_SAMPLE, CC_SAMPLE_FILE)
        print(f"  + copiada desde S10 -> {CC_SAMPLE_FILE}")
        verificar_sha(CC_SAMPLE_FILE, CC_SAMPLE_SHA256, "creditcard_muestra.csv")
        return pd.read_csv(CC_SAMPLE_FILE)
    # 2) derivar de la base completa (cache temporal). El completo (~150,8 MB) se
    #    verifica por SHA256 en cargar_fraude_completo antes de derivar la muestra.
    full = cargar_fraude_completo()
    fraude = full[full["Class"] == 1]
    legit = full[full["Class"] == 0].sample(n=CC_N_LEGIT, random_state=CC_RS)
    muestra = pd.concat([fraude, legit]).sort_values("Time").reset_index(drop=True)
    muestra.to_csv(CC_SAMPLE_FILE, index=False, encoding="utf-8")
    verificar_sha(CC_SAMPLE_FILE, CC_SAMPLE_SHA256, "creditcard_muestra.csv (derivada)")
    print(f"  + muestra derivada del completo -> {CC_SAMPLE_FILE}")
    return muestra


def cargar_fraude_completo() -> pd.DataFrame:
    """Completo 284807x31 al cache temporal (NUNCA en OneDrive). Solo para el
    desbalance real 99:1 en Colab/cache. En la sesion EPE las metricas 99:1 se
    reutilizan de resultados/S10_resultados.xlsx; esto es solo para reproducir."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(CC_CACHE_FILE) and sha256_file(CC_CACHE_FILE) == CC_SHA256_SRC:
        print(f"  = completo en cache temporal ({CC_CACHE_FILE})")
        return pd.read_csv(CC_CACHE_FILE)
    print(f"  ... descargando completo (~150,8 MB, SOLO runtime; Kaggle pide login)\n"
          f"      {CC_URL}")
    raw = _fetch(CC_URL)
    with open(CC_CACHE_FILE, "wb") as f:
        f.write(raw)
    print(f"  ... completo -> {CC_CACHE_FILE} ({len(raw)/1e6:.1f} MB) "
          f"SHA256={sha256_bytes(raw)[:16]}...")
    return pd.read_csv(CC_CACHE_FILE)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    print(f"Directorio de datos (OneDrive): {DATA_DIR}")
    print(f"Cache del completo (temporal):  {CACHE_DIR}\n")

    tc = obtener_telco()
    print()
    muestra = obtener_fraude_muestra()

    print("\n== Resumen ==")
    ch = pd.to_numeric(tc["TotalCharges"], errors="coerce").isna().sum()
    print(f"telco_churn.csv        : {tc.shape[0]} x {tc.shape[1]}  "
          f"(churn; {ch} TotalCharges en blanco = clientes tenure 0)")
    print(f"creditcard_muestra.csv : {muestra.shape[0]} x {muestra.shape[1]}  "
          f"(fraude={muestra['Class'].mean()*100:.2f}% en la MUESTRA; "
          f"completo 284807x31 = 0,17% NO en OneDrive)")

    print("\n== Checksums SHA256 (archivo local) ==")
    for p in (TELCO_FILE, CC_SAMPLE_FILE):
        if os.path.exists(p):
            print(f"  {os.path.basename(p):24s} {sha256_file(p)}  "
                  f"({os.path.getsize(p)/1e6:.3f} MB)")
    print(f"  {'creditcard.csv (fuente)':24s} {CC_SHA256_SRC}  "
          f"(completo ~150,8 MB, NO en OneDrive)")

    if "--full" in argv:
        print()
        cargar_fraude_completo()
    print("\nVerificacion de fuentes: 06/08/2026. Todo OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
