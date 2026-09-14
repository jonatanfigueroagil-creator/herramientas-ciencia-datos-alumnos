# Diccionario de datos — Sesión S06 (Reducción de dimensionalidad)

> Fecha de verificación de fuentes: **18/07/2026**. Generado por `investigador-datos`.
> Todos los datos se regeneran con `descargar_datos.py` (venv del curso). Nada supera
> 25 MB en OneDrive: MNIST se conserva solo como **muestra ≤1 MB**; el completo se
> descarga en runtime y **no** se guarda en disco.
> Los **targets cuantitativos con tolerancia** los define la ficha de la sesión de réplica del paper
> (dueño único). Aquí solo se documentan esquema, procedencia y valores OBSERVADOS.

## Roles de la sesión

| Rol | Dataset | Archivo local | Vía | Esquema |
|---|---|---|---|---|
| Replicación PCA (biplot, convención cov/corr) | **Iris** (Fisher 1936) | `iris.csv` | `sklearn.datasets.load_iris` | 150 × 5 |
| Replicación PCA multivariante / Análisis Factorial | **Wine** (UCI id 109) | `wine.csv` | `sklearn.datasets.load_wine` | 178 × 14 |
| Replicación t-SNE/UMAP (primario, ligero) | **digits 8×8** (UCI Optical Recognition) | `digits.csv` | `sklearn.datasets.load_digits` | 1797 × 65 |
| Replicación t-SNE/UMAP (showcase de alta dim.) | **MNIST_784** (LeCun) | `mnist_muestra.npz` (muestra) | `fetch_openml('mnist_784')` + subsampleo | 2000 × 784 (muestra; completo 70000 × 784) |
| Negocio — PCR (multicolinealidad) | **Communities & Crime** (S05) / **Ames** (S04) | *(se reutiliza)* `../../S05_regularizacion/data/communities.csv` — `../../S04_regresion_multiple/data/AmesHousing.csv` | ya verificados en S04/S05 | 1994 × 128 — 2930 × 82 |

---

## 1. Iris — `iris.csv` (150 × 5)

Medidas morfológicas de 150 flores de lirio (50 por especie). Dataset canónico
para PCA porque ilustra la **convención de estandarización**: con matriz de
covarianza (datos crudos) el PC1 domina por la escala del pétalo; estandarizado
(correlación) el peso se reparte.

| Variable | Tipo | Unidad | Significado |
|---|---|---|---|
| `largo_sepalo_cm` | float | cm | Longitud del sépalo |
| `ancho_sepalo_cm` | float | cm | Ancho del sépalo |
| `largo_petalo_cm` | float | cm | Longitud del pétalo |
| `ancho_petalo_cm` | float | cm | Ancho del pétalo |
| `especie` | categórica | — | `setosa`, `versicolor`, `virginica` (50 c/u) |

- **Procedencia:** Fisher, R.A. (1936). *The use of multiple measurements in taxonomic problems*. Annals of Eugenics 7(2):179–188. Espejado por UCI Machine Learning Repository (Iris) y embebido en scikit-learn.
- **Licencia:** dominio público (UCI, CC BY 4.0 para la copia de UCI).
- **Nota de negocio:** las 4 variables están muy correlacionadas (pétalos) → caso ideal para reducir a 2 componentes sin perder separación de clases.

## 2. Wine — `wine.csv` (178 × 14)

Resultado de 13 análisis químicos de vinos italianos de 3 cultivares. Sirve para
PCA multivariante y para **Análisis Factorial** (variables latentes: cuerpofenólico, color, etc.).

| Variable | Significado (negocio) |
|---|---|
| `alcohol` | Grado alcohólico |
| `malic_acid` | Ácido málico |
| `ash` | Cenizas |
| `alcalinity_of_ash` | Alcalinidad de las cenizas |
| `magnesium` | Magnesio |
| `total_phenols` | Fenoles totales |
| `flavanoids` | Flavonoides (fenoles ligados al sabor) |
| `nonflavanoid_phenols` | Fenoles no flavonoides |
| `proanthocyanins` | Proantocianinas |
| `color_intensity` | Intensidad del color |
| `hue` | Tono (hue) |
| `od280/od315_of_diluted_wines` | Absorbancia OD280/OD315 (proteínas) |
| `proline` | Prolina (aminoácido) |
| `target` | Cultivar de origen: `0` (n=59), `1` (n=71), `2` (n=48) |

- **Procedencia:** Forina, M. et al.; UCI Machine Learning Repository, *Wine* (id 109). Embebido en scikit-learn (`load_wine`).
- **Licencia:** CC BY 4.0 (UCI).
- **Nota:** las 13 variables tienen escalas muy dispares (magnesium ~100, hue ~1) → **estandarizar es obligatorio** antes de PCA/FA.

## 3. digits 8×8 — `digits.csv` (1797 × 65)

Dígitos manuscritos digitalizados a imágenes 8×8 en escala de grises 0–16.
**Primario** para t-SNE/UMAP por ser ligero (se ejecuta en segundos en Colab).

| Variable | Tipo | Rango | Significado |
|---|---|---|---|
| `px_00` … `px_63` | uint8 | 0–16 | Intensidad del píxel (64 = imagen 8×8 aplanada, orden fila-mayor) |
| `digito` | uint8 | 0–9 | Etiqueta del dígito manuscrito |

- **Procedencia:** Alpaydin, E. & Kaynak, C.; UCI *Optical Recognition of Handwritten Digits*. Embebido en scikit-learn (`load_digits`).
- **Licencia:** CC BY 4.0 (UCI).

## 4. MNIST_784 — `mnist_muestra.npz` (muestra 2000 × 784)

Dígitos manuscritos 28×28 (784 píxeles, 0–255). Es el conjunto del **paper de
t-SNE** (van der Maaten & Hinton, 2008) y de **UMAP** (McInnes et al., 2018).

- **Contenido del `.npz`:** `X` (uint8, 2000 × 784) y `y` (uint8, 2000). Muestra **estratificada** de 200 imágenes por dígito, `random_state=42`.
- **Peso:** 0.33 MB comprimido (`np.savez_compressed`) → cumple el límite ≤1 MB de OneDrive.
- **El conjunto COMPLETO (70000 × 784, ~55 MB) NUNCA se guarda en OneDrive.** Se descarga en runtime con `fetch_openml('mnist_784', version=1, as_frame=False)` (cachea en `~/scikit_learn_data`, fuera de OneDrive). Reproducible en Colab con `descargar_datos.py --full-mnist`.
- **Procedencia:** LeCun, Cortes & Burges, *MNIST database*; OpenML id `mnist_784`. Papers de la técnica: van der Maaten & Hinton (2008), JMLR 9; McInnes, Healy & Melville (2018), arXiv:1802.03426.
- **Licencia:** CC BY-SA 3.0 (MNIST original / OpenML).
- **Fallback documentado:** si `fetch_openml` falla o pesa demasiado, el script no genera el `.npz` y avisa que se use **`load_digits` (8×8)** como sustituto del showcase t-SNE/UMAP.

## 5. Negocio (PCR) — datos reutilizados de S04/S05

El laboratorio de PCR (regresión sobre componentes para combatir lamulticolinealidad) **reutiliza** un dataset colineal ya verificado, sin volver a
descargarlo aquí:

- **Communities & Crime** (`../../S05_regularizacion/data/communities.csv`, 1994 × 128): fuerte multicolinealidad entre predictores socioeconómicos → caso natural de PCR/PLS. Verificado en S05 (18/07/2026).
- **Ames Housing** (`../../S04_regresion_multiple/data/AmesHousing.csv`, 2930 × 82): alternativa con colinealidad (p. ej. `Garage Cars`↔`Garage Area`). Verificado en S04 (18/07/2026).

---

## Valores OBSERVADOS en el smoke de viabilidad (18/07/2026)

> Reportados por `descargar_datos.py --smoke`. **No** son los targets oficiales
> (esos viven en la ficha de la sesión de réplica del paper); sirven para confirmar que
> la réplica es reproducible con este venv.

| Prueba | Convención | Valor observado |
|---|---|---|
| PCA Iris — PC1 / PC2 | **Covarianza** (datos crudos) | **92.46% / 5.31%** (PC1+PC2 = 97.77%) |
| PCA Iris — PC1 / PC2 | **Correlación** (estandarizado) | **72.96% / 22.85%** (PC1+PC2 = 95.81%) |
| PCA digits — nº componentes ≥90% var. | Covarianza | **21** de 64 (≥95%: 29; PC1 = 14.89%) |
| Análisis Factorial Iris (varimax, 2 factores) | Estandarizado | Kaiser (autovalores>1): **1**; var. acum. = 93.0% |
| Análisis Factorial Wine (varimax, 3 factores) | Estandarizado | Kaiser (autovalores>1): **3**; var. acum. = 58.4% |
| openTSNE sobre digits (`random_state=42`) | — | OK — embedding (1797, 2) |
| umap-learn sobre digits (`random_state=42`) | — | OK — embedding (1797, 2) |

**Nota de convención (para el notebook):** scikit-learn `PCA` **centra pero no
escala**, por lo que equivale a PCA sobre la **matriz de covarianza**. Para la
convención de **correlación** hay que aplicar `StandardScaler` antes. Los
porcentajes de PC1/PC2 cambian drásticamente entre ambas (ver Iris arriba); el
notebook debe mostrar las dos y fijar cuál usa la réplica.

**Aviso de entorno (blocker resuelto):** `factor_analyzer 0.3.1` (versión quepinneaba la matriz de versiones certificada del curso) **no ejecuta** con `scipy 1.16.3` porque usa
`scipy.sum` (removido en scipy ≥1.14) → `AttributeError`. Se actualizó a
**`factor_analyzer 0.5.1`**, que funciona con numpy 2.3.5 / scipy 1.16.3 sin tocar
los pins de numpy/scipy/pandas/scikit-learn. Debe reflejarse en
la matriz de versiones certificada del curso.
