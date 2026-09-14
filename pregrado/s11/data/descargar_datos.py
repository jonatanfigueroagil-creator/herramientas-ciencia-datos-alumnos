# -*- coding: utf-8 -*-
"""
Descarga y verificación de datasets — Sesión 11 (Series temporales).

Idempotente: si el archivo ya existe y su SHA256 coincide con el esperado,
NO se vuelve a descargar. Funciona en local y en Google Colab.

Ejecución local (usar SIEMPRE la ruta absoluta del venv del curso):
  "C:/Users/Usuario/OneDrive/Cursos/Herramientas de Ciencias de Datos/python" \
      "C:/Users/Usuario/OneDrive/Cursos/Herramientas de Ciencias de Datos/Sesiones/S11_series_temporales/data/descargar_datos.py"

En Colab:
  !pip -q install statsmodels requests
  !python descargar_datos.py

Datasets:
  1) airpassengers.csv   — Box & Jenkins «Serie G» (AirPassengers), 144 meses 1949-01..1960-12.
                           Fuente: statsmodels get_rdataset('AirPassengers','datasets') [Rdatasets];
                           fallback offline: arreglo canónico embebido en este script.
  2) peyton_manning.csv  — ejemplo oficial de Prophet (Taylor & Letham 2018), log page views
                           de la Wikipedia de Peyton Manning. Descarga directa del repo Prophet.
  3) train.csv           — Store Item Demand Forecasting Challenge (Kaggle
                           c/demand-forecasting-kernels-only). Kaggle exige login; se usa un
                           MIRROR ABIERTO (byte-idéntico). ~16.5 MB (<25 MB → se conserva).
     train_muestra.csv   — muestra <1 MB (tienda 1, items 1-10) para portabilidad del notebook.
"""
import io
import os
import sys
import hashlib

try:
    import requests
except ImportError:
    requests = None
import pandas as pd

# --------------------------------------------------------------------------- #
# Rutas relativas al propio script (portable local/Colab)
# --------------------------------------------------------------------------- #
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

F_AIR = os.path.join(DATA_DIR, "airpassengers.csv")
F_PM = os.path.join(DATA_DIR, "peyton_manning.csv")
F_TRAIN = os.path.join(DATA_DIR, "train.csv")
F_SAMPLE = os.path.join(DATA_DIR, "train_muestra.csv")
F_SUMS = os.path.join(DATA_DIR, "CHECKSUMS.sha256")

# --------------------------------------------------------------------------- #
# Fuentes y checksums esperados (verificados 19/07/2026)
# --------------------------------------------------------------------------- #
URL_PM = ("https://raw.githubusercontent.com/facebook/prophet/main/"
          "examples/example_wp_log_peyton_manning.csv")
SHA_PM = "6b1383a6f458e317c5da0488abaf2cadccd7e80c84c7a543d084e1159d5bfa56"

# Mirror abierto del train.csv de Kaggle (primario y secundario byte-idénticos)
URL_TRAIN_PRIMARY = ("https://raw.githubusercontent.com/jgonzalezab/"
                     "Store-Item-Demand-Forecasting/master/Data/train.csv")
URL_TRAIN_SECONDARY = ("https://raw.githubusercontent.com/ugursaricam/"
                       "store-item-demand-forecasting/master/datasets/train.csv")
SHA_TRAIN = "038f25690a65149c94f86ddd3deceda20c037a5cfd754cafdfc539a72992f2ed"

# Arreglo canónico de AirPassengers (Box & Jenkins, Serie G) — fallback offline
AIRPASSENGERS = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
]


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def http_get(url: str, timeout: int = 180) -> bytes:
    if requests is None:
        raise RuntimeError("El paquete 'requests' no está disponible.")
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def already_ok(path: str, expected_sha: str) -> bool:
    return os.path.exists(path) and sha256_file(path) == expected_sha


# --------------------------------------------------------------------------- #
# 1) AirPassengers
# --------------------------------------------------------------------------- #
def build_airpassengers() -> pd.DataFrame:
    """Devuelve DataFrame [month (YYYY-MM), passengers]. Online → Rdatasets;
    si falla, usa el arreglo canónico embebido (offline)."""
    vals = None
    try:
        from statsmodels.datasets import get_rdataset
        ds = get_rdataset("AirPassengers", "datasets", cache=True)
        vals = [int(v) for v in ds.data["value"].values]
        if len(vals) != 144:
            vals = None
    except Exception as exc:  # sin red o sin statsmodels
        print(f"   [aviso] get_rdataset falló ({exc!r}); uso arreglo embebido.")
    if vals is None:
        vals = list(AIRPASSENGERS)
    idx = pd.date_range("1949-01-01", periods=144, freq="MS")
    return pd.DataFrame({"month": idx.strftime("%Y-%m"), "passengers": vals})


def do_airpassengers() -> None:
    df = build_airpassengers()
    assert df.shape == (144, 2), df.shape
    df.to_csv(F_AIR, index=False, encoding="utf-8")
    print(f"   OK airpassengers.csv  {df.shape[0]}x{df.shape[1]}  "
          f"{df.month.iloc[0]}..{df.month.iloc[-1]}  sha={sha256_file(F_AIR)[:12]}")


# --------------------------------------------------------------------------- #
# 2) Peyton Manning
# --------------------------------------------------------------------------- #
def do_peyton() -> None:
    if already_ok(F_PM, SHA_PM):
        print("   OK peyton_manning.csv (ya presente, checksum válido)")
        return
    b = http_get(URL_PM)
    got = sha256_bytes(b)
    if got != SHA_PM:
        print(f"   [aviso] checksum Peyton distinto del esperado ({got[:12]} != "
              f"{SHA_PM[:12]}); el archivo pudo actualizarse en origen.")
    with open(F_PM, "wb") as fh:
        fh.write(b)
    df = pd.read_csv(F_PM)
    assert list(df.columns) == ["ds", "y"], df.columns
    print(f"   OK peyton_manning.csv  {df.shape[0]}x{df.shape[1]}  "
          f"{df.ds.min()}..{df.ds.max()}  sha={sha256_file(F_PM)[:12]}")


# --------------------------------------------------------------------------- #
# 3) Store Item Demand (train.csv) + muestra
# --------------------------------------------------------------------------- #
def do_train() -> None:
    if already_ok(F_TRAIN, SHA_TRAIN):
        print("   OK train.csv (ya presente, checksum válido)")
    else:
        b = None
        for url in (URL_TRAIN_PRIMARY, URL_TRAIN_SECONDARY):
            try:
                b = http_get(url)
                if sha256_bytes(b) == SHA_TRAIN:
                    print(f"   descargado desde: {url}")
                    break
                print(f"   [aviso] checksum distinto desde {url}; pruebo otro mirror.")
                b = None
            except Exception as exc:
                print(f"   [aviso] falló {url}: {exc!r}")
        if b is None:
            raise RuntimeError(
                "No se pudo descargar train.csv de ningún mirror. "
                "Ver data/README_datos.md para la descarga manual desde Kaggle.")
        with open(F_TRAIN, "wb") as fh:
            fh.write(b)

    df = pd.read_csv(F_TRAIN)
    assert df.shape == (913000, 4), df.shape
    assert list(df.columns) == ["date", "store", "item", "sales"], df.columns
    print(f"   OK train.csv  {df.shape[0]}x{df.shape[1]}  "
          f"{df.date.min()}..{df.date.max()}  {df.store.nunique()} tiendas x "
          f"{df.item.nunique()} items  sha={sha256_file(F_TRAIN)[:12]}")

    # muestra <1 MB: tienda 1, items 1-10 (10 series diarias completas 2013-2017)
    sub = df[(df.store == 1) & (df.item <= 10)].reset_index(drop=True)
    sub.to_csv(F_SAMPLE, index=False, encoding="utf-8")
    size_kb = os.path.getsize(F_SAMPLE) / 1024
    print(f"   OK train_muestra.csv  {sub.shape[0]}x{sub.shape[1]}  "
          f"{size_kb:.0f} KB  sha={sha256_file(F_SAMPLE)[:12]}")


# --------------------------------------------------------------------------- #
# Checksums
# --------------------------------------------------------------------------- #
def write_checksums() -> None:
    lines = []
    for path in (F_AIR, F_PM, F_TRAIN, F_SAMPLE):
        if os.path.exists(path):
            lines.append(f"{sha256_file(path)}  {os.path.basename(path)}")
    with open(F_SUMS, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("   CHECKSUMS.sha256 actualizado")


def main() -> int:
    print("== Sesión 11, descarga de datos ==")
    print(f"Python: {sys.version.split()[0]}  |  destino: {DATA_DIR}")
    print("[1/3] AirPassengers")
    do_airpassengers()
    print("[2/3] Peyton Manning")
    do_peyton()
    print("[3/3] Store Item Demand (train.csv + muestra)")
    do_train()
    write_checksums()
    print("Listo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
