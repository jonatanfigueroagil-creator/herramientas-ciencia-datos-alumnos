# -*- coding: utf-8 -*-
"""
descargar_datos.py — Sesion 12 (Inferencia causal)
Curso "Herramientas para la Ciencia de Datos" — UPC.

Materializa los tres datasets de la sesion:
  1) card_krueger_1994.csv   -> DiD (salario minimo NJ/PA). Fuente: David Card.
  2) lalonde_nsw.csv         -> PSM, muestra EXPERIMENTAL NSW (Dehejia-Wahba). causaldata.
     lalonde_cps_obs.csv     -> comparacion OBSERVACIONAL CPS-1. causaldata.
  3) retencion_observacional.csv -> caso de negocio SIMULADO (DGP con semilla fija).

Idempotente: si el CSV existe y su SHA256 coincide con CHECKSUMS.txt, no lo regenera.
Funciona en local y en Colab. Ejecutar SIEMPRE con el python del venv (ruta absoluta):

    "C:/Users/Usuario/OneDrive/Cursos/Herramientas de Ciencias de Datos/python" \
        "C:/Users/Usuario/OneDrive/Cursos/Herramientas de Ciencias de Datos/Sesiones/S12_inferencia_causal/data/descargar_datos.py"

Verificado el 19/07/2026. NO deja archivos > 25 MB en OneDrive (los tres CSV son pequenos).
"""
import os
import io
import sys
import zipfile
import hashlib

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rutas: la carpeta de este script (funciona en local y si se sube a Colab)
# ---------------------------------------------------------------------------
try:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # pegado en una celda de Colab
    DATA_DIR = os.getcwd()

CK_URL = "https://davidcard.berkeley.edu/data_sets/njmin.zip"
CK_ZIP_SHA256 = "41a29aec6b489658b20030d044a2b20a5f5a04b871cde41df057afd6013ba587"

CHECKSUMS = os.path.join(DATA_DIR, "CHECKSUMS.txt")


# ---------------------------------------------------------------------------
# Utilidades de checksum / idempotencia
# ---------------------------------------------------------------------------
def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _leer_checksums() -> dict:
    reg = {}
    if os.path.exists(CHECKSUMS):
        with open(CHECKSUMS, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea or linea.startswith("#"):
                    continue
                partes = linea.split(None, 1)
                if len(partes) == 2:
                    reg[partes[1]] = partes[0]
    return reg


def _escribir_checksums(reg: dict) -> None:
    with open(CHECKSUMS, "w", encoding="utf-8") as f:
        f.write("# SHA256 de los datasets de la Sesion 12 (generado por descargar_datos.py)\n")
        f.write("# formato: <sha256>  <archivo>\n")
        for nombre in sorted(reg):
            f.write(f"{reg[nombre]}  {nombre}\n")


def _ya_vigente(nombre: str, reg: dict) -> bool:
    """True si el archivo existe y su SHA256 coincide con el registrado."""
    ruta = os.path.join(DATA_DIR, nombre)
    if not os.path.exists(ruta):
        return False
    esperado = reg.get(nombre)
    if esperado is None:
        return False
    return _sha256_file(ruta) == esperado


def _materializar(df: pd.DataFrame, nombre: str, reg: dict) -> None:
    ruta = os.path.join(DATA_DIR, nombre)
    df.to_csv(ruta, index=False, encoding="utf-8")
    reg[nombre] = _sha256_file(ruta)
    mb = os.path.getsize(ruta) / (1024 * 1024)
    print(f"  [OK] {nombre:<28} {df.shape[0]:>6} x {df.shape[1]:<2}  {mb:6.2f} MB  sha256={reg[nombre][:12]}...")


# ---------------------------------------------------------------------------
# 1) Card & Krueger (1994) — DiD salario minimo NJ/PA
# ---------------------------------------------------------------------------
CK_COLS = [
    "SHEET", "CHAIN", "CO_OWNED", "STATE", "SOUTHJ", "CENTRALJ", "NORTHJ", "PA1",
    "PA2", "SHORE", "NCALLS", "EMPFT", "EMPPT", "NMGRS", "WAGE_ST", "INCTIME",
    "FIRSTINC", "BONUS", "PCTAFF", "MEAL", "OPEN", "HRSOPEN", "PSODA", "PFRY",
    "PENTREE", "NREGS", "NREGS11", "TYPE2", "STATUS2", "DATE2", "NCALLS2",
    "EMPFT2", "EMPPT2", "NMGRS2", "WAGE_ST2", "INCTIME2", "FIRSTIN2", "SPECIAL2",
    "MEALS2", "OPEN2R", "HRSOPEN2", "PSODA2", "PFRY2", "PENTREE2", "NREGS2",
    "NREGS112",
]


def _descargar_bytes(url: str) -> bytes:
    import requests
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    return r.content


def cargar_card_krueger() -> pd.DataFrame:
    """Descarga njmin.zip de David Card, parsea public.dat (410 x 46) y anade FTE."""
    contenido = _descargar_bytes(CK_URL)
    got = _sha256_bytes(contenido)
    if got != CK_ZIP_SHA256:
        print(f"  [AVISO] SHA256 del ZIP cambio (esperado {CK_ZIP_SHA256[:12]}..., "
              f"obtenido {got[:12]}...). Se continua, pero revisar la fuente.")
    z = zipfile.ZipFile(io.BytesIO(contenido))
    raw = z.read("public.dat").decode("latin-1")
    # public.dat usa entrada tipo lista de SAS: separado por espacios, faltantes = "."
    df = pd.read_csv(io.StringIO(raw), sep=r"\s+", header=None,
                     names=CK_COLS, na_values=".")
    # FTE = empleados de tiempo completo + gerentes + 0.5 * medio tiempo (definicion del paper)
    df["FTE"] = df["EMPPT"] * 0.5 + df["EMPFT"] + df["NMGRS"]
    df["FTE2"] = df["EMPPT2"] * 0.5 + df["EMPFT2"] + df["NMGRS2"]
    df["DEMP"] = df["FTE2"] - df["FTE"]
    df["NJ"] = df["STATE"]  # alias legible: 1 = New Jersey, 0 = Pennsylvania
    return df


# ---------------------------------------------------------------------------
# 2) LaLonde / NSW (Dehejia-Wahba) — PSM
# ---------------------------------------------------------------------------
def cargar_nsw():
    """Devuelve (nsw_experimental, cps_observacional) desde causaldata."""
    from causaldata import nsw_mixtape, cps_mixtape
    nsw = nsw_mixtape.load_pandas().data.copy()   # 445 x 11 (185 tratados + 260 control)
    cps = cps_mixtape.load_pandas().data.copy()   # 15992 x 11 (todos control, CPS-1)
    return nsw, cps


# ---------------------------------------------------------------------------
# 3) Retencion (negocio) — DGP SIMULADO con efecto verdadero conocido
# ---------------------------------------------------------------------------
# DUENO UNICO DEL DGP: la ficha de réplica del paper (reconciliado 19/07/2026).
# Semilla, tamano y parametros EXACTOS de esa seccion: default_rng(20260719), N=5000,
# TRUE_ATE=8,0, value = 50 + 8*campaign + 20*engagement + 3*tenure + N(0,8).
# `engagement` es CONFUSOR (engagement -> campaign y engagement -> value);
# `tenure` es covariable de PRECISION (tenure -> value, no -> campaign).
RETENCION_SEED = 20260719
RETENCION_N = 5000
RETENCION_TRUE_ATE = 8.0  # efecto verdadero (constante) -> ATE = ATT = 8,0 USD


def generar_retencion(n=RETENCION_N, seed=RETENCION_SEED,
                      true_ate=RETENCION_TRUE_ATE) -> pd.DataFrame:
    """
    Campana de retencion sobre datos OBSERVACIONALES (no aleatorizados).
    DGP dueno unico: la ficha de réplica del paper El equipo de marketing
    eligio a quien contactar mirando la ACTIVIDAD PREVIA del cliente (`engagement`):
    los mas activos fueron los mas contactados. `engagement` tambien predice el VALOR
    futuro, de modo que es un CONFUSOR: el estimador naive queda SESGADO AL ALZA
    (los clientes ya buenos son los mas contactados) y solo el ajuste por el conjunto
    de puerta trasera {engagement} recupera el efecto verdadero (ATE = 8,0 USD).
    `tenure` (antiguedad, meses) afecta al valor pero NO a la asignacion -> precision.
    """
    rng = np.random.default_rng(seed)
    engagement = rng.normal(0, 1, n)                    # confusor (actividad previa, z-score)
    tenure = rng.normal(24, 6, n)                       # antiguedad en meses (precision)
    # Asignacion NO aleatoria: los clientes mas activos son mas contactados
    p_treat = 1.0 / (1.0 + np.exp(-(0.9 * engagement - 0.3)))
    campaign = rng.binomial(1, p_treat)                 # 1 = recibio campana (TRATO)
    # Resultado: valor incremental del cliente (USD del proximo trimestre)
    value = (50.0 + true_ate * campaign + 20.0 * engagement + 3.0 * tenure
             + rng.normal(0, 8, n))

    return pd.DataFrame({
        "engagement": engagement,   # confusor (actividad previa del cliente, z-score)
        "tenure": tenure,           # antiguedad en meses (covariable de precision)
        "campaign": campaign,       # 1 = recibio la campana de retencion (TRATO)
        "value": value,             # valor incremental del cliente en USD (RESULTADO)
    })


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------
def main():
    print(f"Sesion 12 — materializando datasets en:\n  {DATA_DIR}\n")
    reg = _leer_checksums()

    # 1) Card & Krueger
    if _ya_vigente("card_krueger_1994.csv", reg):
        print("  [skip] card_krueger_1994.csv ya vigente (checksum OK)")
    else:
        try:
            ck = cargar_card_krueger()
            assert ck.shape == (410, 50), f"forma inesperada CK: {ck.shape}"
            _materializar(ck, "card_krueger_1994.csv", reg)
        except Exception as e:
            print(f"  [ERROR] Card & Krueger no se pudo materializar: {type(e).__name__}: {e}")

    # 2) NSW experimental + CPS observacional
    faltan_nsw = not (_ya_vigente("lalonde_nsw.csv", reg) and _ya_vigente("lalonde_cps_obs.csv", reg))
    if not faltan_nsw:
        print("  [skip] lalonde_nsw.csv y lalonde_cps_obs.csv ya vigentes (checksum OK)")
    else:
        try:
            nsw, cps = cargar_nsw()
            assert nsw.shape[0] == 445 and cps.shape[0] == 15992, \
                f"formas NSW/CPS inesperadas: {nsw.shape} / {cps.shape}"
            _materializar(nsw, "lalonde_nsw.csv", reg)
            _materializar(cps, "lalonde_cps_obs.csv", reg)
        except Exception as e:
            print(f"  [ERROR] NSW/CPS no se pudo materializar: {type(e).__name__}: {e}")

    # 3) Retencion (simulada — dueno del DGP: REPLICACION_PAPER.md Sección 5)
    if _ya_vigente("retencion_observacional.csv", reg):
        print("  [skip] retencion_observacional.csv ya vigente (checksum OK)")
    else:
        ret = generar_retencion()
        _materializar(ret, "retencion_observacional.csv", reg)
        print(f"         (DGP REPLICACION Sección 5, semilla={RETENCION_SEED}, N={RETENCION_N}, "
              f"efecto verdadero ATE={RETENCION_TRUE_ATE} USD)")

    _escribir_checksums(reg)
    print(f"\nChecksums -> {CHECKSUMS}")
    print("Listo.")


if __name__ == "__main__":
    sys.exit(main())
