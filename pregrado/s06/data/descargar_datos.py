# -*- coding: utf-8 -*-
"""
Descarga y verificacion de los datos de la Sesion S06
(Reduccion de dimensionalidad: PCA + Analisis Factorial + t-SNE + UMAP)
del curso UPC "Herramientas para la Ciencia de Datos".

Cuatro conjuntos, todos ligeros o subsampleados (nada >25 MB queda en OneDrive):

  1) IRIS (Fisher 1936)  -> "iris.csv"   150 x 5
     REPLICACION del PCA (biplot, varianza explicada). Integrado en sklearn
     (sklearn.datasets.load_iris). Cita: Fisher, R.A. (1936). "The use of
     multiple measurements in taxonomic problems". Annals of Eugenics 7(2).
     Origen sklearn: UCI Iris. Ancla la convencion de estandarizacion:
     covarianza (datos crudos) vs correlacion (estandarizados) cambian los %.

  2) WINE  (Forina / UCI id 109)  -> "wine.csv"   178 x 14
     REPLICACION del PCA multivariante y del Analisis Factorial (13 medidas
     quimicas). Integrado en sklearn (sklearn.datasets.load_wine).

  3) DIGITS (8x8, Alpaydin & Kaydir; UCI Optical Recognition) -> "digits.csv"
     1797 x 65 (64 pixeles 0-16 + digito 0-9). Integrado en sklearn
     (sklearn.datasets.load_digits). PRIMARIO y ligero para t-SNE / UMAP y para
     el conteo de componentes al 90% de varianza.

  4) MNIST_784 (LeCun; van der Maaten-Hinton 2008; McInnes 2018)
     -> "mnist_muestra.npz"  (MUESTRA estratificada 2000 x 784, uint8, <1 MB)
     El completo es 70000 x 784 (~55 MB en memoria): NUNCA se guarda en OneDrive.
     Se descarga con fetch_openml('mnist_784', as_frame=False) SOLO en runtime,
     se toma una muestra estratificada (200 por digito, random_state=42) y se
     guarda comprimida (np.savez_compressed, uint8). El notebook usa esta muestra;
     quien quiera el completo lo baja en Colab con la funcion de este script.
     Fallback si fetch_openml falla o pesa demasiado: usar load_digits (8x8)
     como sustituto del showcase t-SNE/UMAP (queda documentado y avisado).

Idempotente: si un archivo ya existe y su esquema (filas x columnas) es correcto,
NO se vuelve a generar. Funciona en local (Windows/venv) y en Google Colab.

Uso:
    "<python_del_venv>" descargar_datos.py            # genera + verifica esquema
    "<python_del_venv>" descargar_datos.py --smoke     # ademas corre el smoke de viabilidad
    "<python_del_venv>" descargar_datos.py --full-mnist # baja MNIST completo (no lo guarda)
    # o en Colab:  %run descargar_datos.py

Requiere factor_analyzer >= 0.5.0 para el smoke del Analisis Factorial (la 0.3.1
usa scipy.sum, removido en scipy >= 1.14 -> AttributeError).

Fecha de verificacion de fuentes: 18/07/2026.
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
if "google.colab" in sys.modules:
    DATA_DIR = os.getcwd()
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

IRIS_FILE = os.path.join(DATA_DIR, "iris.csv")            # 150 x 5
WINE_FILE = os.path.join(DATA_DIR, "wine.csv")            # 178 x 14
DIGITS_FILE = os.path.join(DATA_DIR, "digits.csv")        # 1797 x 65
MNIST_FILE = os.path.join(DATA_DIR, "mnist_muestra.npz")  # 2000 x 784 (uint8)

MNIST_POR_CLASE = 200          # 200 x 10 digitos = 2000 imagenes
MNIST_RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _csv_valido(path, nrows, ncols) -> bool:
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
        if df.shape != (nrows, ncols):
            print(f"  ! {os.path.basename(path)} esquema {df.shape} != "
                  f"({nrows}, {ncols}); se regenera.")
            return False
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ! {os.path.basename(path)} no valida ({e}); se regenera.")
        return False


def _npz_valido(path, nrows, ncols) -> bool:
    if not os.path.exists(path):
        return False
    try:
        d = np.load(path)
        X, y = d["X"], d["y"]
        if X.shape != (nrows, ncols) or y.shape[0] != nrows:
            print(f"  ! {os.path.basename(path)} esquema {X.shape} != "
                  f"({nrows}, {ncols}); se regenera.")
            return False
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ! {os.path.basename(path)} no valida ({e}); se regenera.")
        return False


# ---------------------------------------------------------------------------
# 1) IRIS  (replicacion PCA)
# ---------------------------------------------------------------------------
def obtener_iris() -> pd.DataFrame:
    print("== IRIS 150x5 (Fisher 1936; replicacion PCA / convencion cov vs corr) ==")
    if _csv_valido(IRIS_FILE, 150, 5):
        print(f"  = presente y verificado: {IRIS_FILE}")
        return pd.read_csv(IRIS_FILE)
    from sklearn.datasets import load_iris
    d = load_iris(as_frame=True)
    df = d.frame.rename(columns={
        "sepal length (cm)": "largo_sepalo_cm",
        "sepal width (cm)": "ancho_sepalo_cm",
        "petal length (cm)": "largo_petalo_cm",
        "petal width (cm)": "ancho_petalo_cm",
        "target": "especie_cod",
    })
    df["especie"] = pd.Categorical.from_codes(df["especie_cod"], d.target_names)
    df = df.drop(columns=["especie_cod"])
    df.to_csv(IRIS_FILE, index=False, encoding="utf-8")
    print(f"  + generado via load_iris -> {IRIS_FILE}")
    return df


# ---------------------------------------------------------------------------
# 2) WINE  (replicacion PCA / Analisis Factorial)
# ---------------------------------------------------------------------------
def obtener_wine() -> pd.DataFrame:
    print("== WINE 178x14 (UCI id 109; replicacion PCA / Analisis Factorial) ==")
    if _csv_valido(WINE_FILE, 178, 14):
        print(f"  = presente y verificado: {WINE_FILE}")
        return pd.read_csv(WINE_FILE)
    from sklearn.datasets import load_wine
    d = load_wine(as_frame=True)
    df = d.frame  # 13 features + target
    df.to_csv(WINE_FILE, index=False, encoding="utf-8")
    print(f"  + generado via load_wine -> {WINE_FILE}")
    return df


# ---------------------------------------------------------------------------
# 3) DIGITS 8x8  (primario t-SNE / UMAP, ligero)
# ---------------------------------------------------------------------------
def obtener_digits() -> pd.DataFrame:
    print("== DIGITS 1797x65 (UCI Optical Recognition 8x8; primario t-SNE/UMAP) ==")
    if _csv_valido(DIGITS_FILE, 1797, 65):
        print(f"  = presente y verificado: {DIGITS_FILE}")
        return pd.read_csv(DIGITS_FILE)
    from sklearn.datasets import load_digits
    d = load_digits()
    cols = [f"px_{i:02d}" for i in range(64)]
    df = pd.DataFrame(d.data.astype(np.uint8), columns=cols)
    df["digito"] = d.target.astype(np.uint8)
    df.to_csv(DIGITS_FILE, index=False, encoding="utf-8")
    print(f"  + generado via load_digits -> {DIGITS_FILE}")
    return df


# ---------------------------------------------------------------------------
# 4) MNIST_784  (muestra estratificada <1 MB; el completo NUNCA en OneDrive)
# ---------------------------------------------------------------------------
def _descargar_mnist_completo():
    """Descarga MNIST 70000x784 SOLO en runtime. Devuelve (X uint8, y int).

    fetch_openml cachea en ~/scikit_learn_data (fuera de OneDrive). Este arreglo
    completo NO se guarda en la carpeta del curso.
    """
    from sklearn.datasets import fetch_openml
    print("  ... descargando MNIST_784 completo via fetch_openml (solo runtime)")
    d = fetch_openml("mnist_784", version=1, as_frame=False)
    X = d.data.astype(np.uint8)          # 70000 x 784, 0-255
    y = d.target.astype(np.int16)        # etiquetas '0'..'9' -> int
    print(f"  ... MNIST completo cargado: {X.shape} (NO se guarda en disco)")
    return X, y


def obtener_mnist_muestra():
    """Devuelve (X, y) de la MUESTRA estratificada 2000x784. La genera si falta."""
    n_total = MNIST_POR_CLASE * 10
    print(f"== MNIST muestra {n_total}x784 (LeCun; van der Maaten-Hinton 2008) ==")
    if _npz_valido(MNIST_FILE, n_total, 784):
        print(f"  = presente y verificado: {MNIST_FILE}")
        d = np.load(MNIST_FILE)
        return d["X"], d["y"]

    try:
        X, y = _descargar_mnist_completo()
    except Exception as e:  # noqa: BLE001
        print(f"  ! fetch_openml('mnist_784') FALLO ({e!r}).")
        print("  ! FALLBACK: usar load_digits (8x8) como sustituto del showcase "
              "t-SNE/UMAP. No se genera mnist_muestra.npz.")
        return None, None

    rng = np.random.RandomState(MNIST_RANDOM_STATE)
    idx_sel = []
    for c in range(10):
        idx_c = np.where(y == c)[0]
        idx_sel.append(rng.choice(idx_c, size=MNIST_POR_CLASE, replace=False))
    idx_sel = np.concatenate(idx_sel)
    rng.shuffle(idx_sel)
    Xs = X[idx_sel].astype(np.uint8)
    ys = y[idx_sel].astype(np.uint8)
    np.savez_compressed(MNIST_FILE, X=Xs, y=ys)
    mb = os.path.getsize(MNIST_FILE) / 1e6
    print(f"  + muestra estratificada ({MNIST_POR_CLASE}/clase, rs="
          f"{MNIST_RANDOM_STATE}) -> {MNIST_FILE} ({mb:.2f} MB)")
    if mb > 1.0:
        print(f"  ! AVISO: la muestra pesa {mb:.2f} MB (>1 MB). Reducir "
              f"MNIST_POR_CLASE si debe mantenerse <=1 MB en OneDrive.")
    return Xs, ys


# ---------------------------------------------------------------------------
# SMOKE de VIABILIDAD (reporta valores OBSERVADOS; NO define targets)
# ---------------------------------------------------------------------------
def smoke():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    print("\n########## SMOKE DE VIABILIDAD S06 ##########")

    # -- PCA sobre IRIS: convencion COVARIANZA (crudo) vs CORRELACION (estand.) --
    print("\n== PCA IRIS: varianza explicada PC1/PC2 en AMBAS convenciones ==")
    iris = obtener_iris()
    Xi = iris[["largo_sepalo_cm", "ancho_sepalo_cm",
               "largo_petalo_cm", "ancho_petalo_cm"]].values

    pca_cov = PCA().fit(Xi)  # sklearn centra pero NO escala -> matriz de COVARIANZA
    vr_cov = pca_cov.explained_variance_ratio_
    print("  [COVARIANZA / datos crudos]  "
          f"PC1={vr_cov[0]*100:.2f}%  PC2={vr_cov[1]*100:.2f}%  "
          f"(PC1+PC2={ (vr_cov[0]+vr_cov[1])*100:.2f}%)")

    Xi_z = StandardScaler().fit_transform(Xi)
    pca_corr = PCA().fit(Xi_z)  # sobre estandarizados -> matriz de CORRELACION
    vr_corr = pca_corr.explained_variance_ratio_
    print("  [CORRELACION / estandarizado] "
          f"PC1={vr_corr[0]*100:.2f}%  PC2={vr_corr[1]*100:.2f}%  "
          f"(PC1+PC2={ (vr_corr[0]+vr_corr[1])*100:.2f}%)")

    # -- num de componentes para >=90% de varianza en DIGITS --
    print("\n== DIGITS: num de componentes para >=90% de varianza ==")
    digits = obtener_digits()
    Xd = digits[[c for c in digits.columns if c.startswith("px_")]].values.astype(float)
    pca_d = PCA().fit(Xd)  # covarianza (todos los pixeles 0-16, misma escala)
    cum = np.cumsum(pca_d.explained_variance_ratio_)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    print(f"  [COVARIANZA]  num comps >=90%: {n90}  (>=95%: {n95})  de 64  |  "
          f"PC1={pca_d.explained_variance_ratio_[0]*100:.2f}%")

    # -- Analisis Factorial (factor_analyzer, rotacion varimax) sobre IRIS y WINE --
    print("\n== factor_analyzer (varimax) sobre IRIS y WINE ==")
    try:
        from factor_analyzer import FactorAnalyzer
        fa_i = FactorAnalyzer(n_factors=2, rotation="varimax")
        fa_i.fit(StandardScaler().fit_transform(Xi))
        ev_i, _ = fa_i.get_eigenvalues()
        var_i = fa_i.get_factor_variance()  # (varianza, prop, prop_acumulada)
        print(f"  IRIS  eigenvalues>1 (Kaiser): {int((ev_i > 1).sum())}  |  "
              f"var. acum. 2 factores varimax={var_i[2][-1]*100:.1f}%")
        wine = obtener_wine()
        Xw = wine[[c for c in wine.columns if c != "target"]].values
        fa_w = FactorAnalyzer(n_factors=3, rotation="varimax")
        fa_w.fit(StandardScaler().fit_transform(Xw))
        ev_w, _ = fa_w.get_eigenvalues()
        var_w = fa_w.get_factor_variance()
        print(f"  WINE  eigenvalues>1 (Kaiser): {int((ev_w > 1).sum())}  |  "
              f"var. acum. 3 factores varimax={var_w[2][-1]*100:.1f}%")
        import factor_analyzer as _fa
        print(f"  factor_analyzer: OK (v{getattr(_fa, '__version__', '?')})")
    except Exception as e:  # noqa: BLE001
        print(f"  factor_analyzer: FALLA -> {e!r}")

    # -- t-SNE (openTSNE) sobre load_digits, random_state fijo --
    print("\n== openTSNE sobre load_digits (random_state=42) ==")
    try:
        from openTSNE import TSNE as OpenTSNE
        emb = OpenTSNE(n_components=2, perplexity=30, random_state=42,
                       n_jobs=1, verbose=False).fit(Xd)
        emb = np.asarray(emb)
        print(f"  openTSNE: OK  embedding {emb.shape}  "
              f"rango x=[{emb[:,0].min():.1f},{emb[:,0].max():.1f}]")
    except Exception as e:  # noqa: BLE001
        print(f"  openTSNE: FALLA -> {e!r}")
        print("  -> fallback sugerido: sklearn.manifold.TSNE")

    # -- UMAP (umap-learn) sobre load_digits, random_state fijo --
    print("\n== umap-learn sobre load_digits (random_state=42) ==")
    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                            random_state=42)
        emb_u = reducer.fit_transform(Xd)
        print(f"  umap-learn: OK  embedding {emb_u.shape}  "
              f"rango x=[{emb_u[:,0].min():.1f},{emb_u[:,0].max():.1f}]")
    except Exception as e:  # noqa: BLE001
        print(f"  umap-learn: FALLA -> {e!r}")
        print("  -> fallback sugerido: sklearn.manifold.TSNE")

    print("\n  (Los targets con tolerancia los fija investigador-tema en "
          "la ficha de réplica del paper; aqui solo se OBSERVAN.)")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    print(f"Directorio de datos: {DATA_DIR}\n")

    iris = obtener_iris()
    print()
    wine = obtener_wine()
    print()
    digits = obtener_digits()
    print()
    Xm, ym = obtener_mnist_muestra()

    print("\n== Resumen ==")
    print(f"iris.csv          : {iris.shape[0]} x {iris.shape[1]}")
    print(f"wine.csv          : {wine.shape[0]} x {wine.shape[1]}")
    print(f"digits.csv        : {digits.shape[0]} x {digits.shape[1]}")
    if Xm is not None:
        print(f"mnist_muestra.npz : {Xm.shape[0]} x {Xm.shape[1]} (muestra; completo NO en disco)")
    else:
        print("mnist_muestra.npz : NO generada (fetch_openml fallo; usar load_digits)")

    print("\n== Checksums SHA256 ==")
    for p in (IRIS_FILE, WINE_FILE, DIGITS_FILE, MNIST_FILE):
        if os.path.exists(p):
            print(f"  {os.path.basename(p):18s} {sha256_file(p)}")

    if "--full-mnist" in argv:
        _descargar_mnist_completo()
    if "--smoke" in argv:
        smoke()
    print("\nVerificacion de fuentes: 18/07/2026. Todo OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
