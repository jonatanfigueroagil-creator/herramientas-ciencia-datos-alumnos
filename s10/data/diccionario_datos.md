# Diccionario de datos — Sesión S10 (Clasificadores clásicos)

> Verificación real de fuentes: **19/07/2026** (Ola A, datos primero). Toda la
> carga es programática con el venv del curso; no se dio por verificado ningún
> dataset sin cargarlo y contar filas × columnas. Genera y valida los archivos
> `descargar_datos.py`.
>
> **Regla de peso:** el CSV completo de fraude pesa ~150.8 MB (>25 MB) → **NUNCA
> en OneDrive**. Solo se versionan `descargar_datos.py`, este diccionario,
> `README_datos.md` y la **muestra ≤1 MB** (`creditcard_muestra.csv`).

---

## 1. Iris — REPLICACIÓN LDA (Fisher, 1936)

- **Rol:** replicación del paper seminal del análisis discriminante lineal.
- **Paper:** Fisher, R. A. (1936). *The Use of Multiple Measurements in Taxonomic
 Problems*. Annals of Eugenics 7(2), 179–188. Datos recolectados por Anderson, E.
 (1935), península de Gaspé.
- **Fuente primaria (offline):** `sklearn.datasets.load_iris` (dataset integrado, determinista, sin red) → materializado como `iris.csv`.
- **Fuente del sílabo / verificación cruzada:** UCI id 53
 <https://archive.ics.uci.edu/dataset/53/iris> (endpoints verificados el
 19/07/2026: `.../static/public/53/iris.zip` HTTP 200 con `iris.data`,
 `bezdekIris.data`, `iris.names`; y `.../ml/machine-learning-databases/iris/iris.data`
 HTTP 200).
- **Esquema:** 150 filas × 5 columnas. 3 clases balanceadas (50 flores por especie).
- **Archivo local:** `iris.csv` (~0.004 MB; regenerable desde sklearn — el
 `.gitignore` del curso no lo versiona por ser regenerable, pero se conserva en
 disco porque es liviano).
- **SHA256 (archivo local generado):** `d32f10723d055c863a7e257d933706ec4fb6e9ec60e9567f68e377bbc6ca9da9`
- **Licencia:** dominio público (dataset clásico de libro de texto).

| Variable | Tipo | Unidad | Significado |
|---|---|---|---|
| `sepal_length_cm` | float | cm | Longitud del sépalo |
| `sepal_width_cm` | float | cm | Ancho del sépalo |
| `petal_length_cm` | float | cm | Longitud del pétalo (variable más discriminante) |
| `petal_width_cm` | float | cm | Ancho del pétalo (variable más discriminante) |
| `species` | categórica | — | Especie objetivo: `setosa`, `versicolor`, `virginica` |

**Nota de procedencia (discrepancia conocida de Iris).** `sklearn.load_iris`
coincide con la versión **corregida** de Fisher (`bezdekIris.data`). Difiere de la
copia histórica `iris.data` de UCI en **exactamente 2 filas** (diferencia máxima0.5 cm en una sola medida) — verificado el 19/07/2026 con el cruce
`--verify-uci`. La discrepancia no altera la frontera LDA/QDA. Se toma la versión
sklearn como canónica del curso.

---

## 2. Credit Card Fraud (ULB) — NEGOCIO (desbalance 99:1)

- **Rol:** caso de negocio de clasificación con clases severamente desbalanceadas
 (detección de fraude), para ilustrar la paradoja de la accuracy, ROC-AUC vs.
 PR-AUC y el efecto de SMOTE.
- **Origen académico:** Dal Pozzolo, A., Caelen, O., Johnson, R. A. & Bontempi, G.
 (2015). *Calibrating Probability with Undersampling for Unbalanced
 Classification*. IEEE Symposium Series on Computational Intelligence (SSCI).
 Colaboración Worldline – Machine Learning Group (MLG), Université Libre de
 Bruxelles (ULB).
- **Fuente del sílabo:** Kaggle `mlg-ulb/creditcardfraud` → **requiere login**
 (decisión del curso 17/07/2026: sin token de API).
- **Fallback aplicado (mirror abierto, resuelto 19/07/2026):** HuggingFace
 `David-Egea/Creditcard-fraud-detection` →
 `.../resolve/main/creditcard.csv`. Descarga real validada: 150.83 MB,
 Content-Type `text/csv`, cabecera `Time,V1..V28,Amount,Class`. Contenido
 **idéntico** al CSV oficial de Kaggle.
- **Esquema (completo):** 284 807 filas × 31 columnas. **492 fraudes** (`Class=1`)
 = **0.1727 %** (ratio ≈ **99.83: 0.17**, es decir ~578:1). **0 nulos**.
- **SHA256 (fuente completa):** `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`
- **Peso:** completo ~150.8 MB → **NUNCA en OneDrive**. Se descarga en runtime al
 caché temporal `%TEMP%/curso_upc_s10_creditcard/creditcard.csv` (fuera de OneDrive) y no se conserva en la carpeta del curso.
- **Licencia:** Database Contents License (DbCL); uso libre con crédito a
 ULB/Worldline.

| Variable | Tipo | Significado de negocio |
|---|---|---|
| `Time` | float (s) | Segundos transcurridos desde la primera transacción del período (0 – 172 792, ≈ 48 h) |
| `V1` … `V28` | float | 28 componentes de PCA **anonimizadas** por confidencialidad (features latentes del comportamiento transaccional; no interpretables individualmente) |
| `Amount` | float | Importe de la transacción (0 – 25 691.16). Útil para análisis sensible al costo |
| `Class` | binaria {0,1} | **Objetivo:** 0 = transacción legítima, 1 = **fraude** (clase minoritaria) |

### Muestra de laboratorio — `creditcard_muestra.csv`

- **Diseño:** TODOS los **492 fraudes** + submuestra aleatoria de **1350**
 transacciones legítimas (`random_state=42`), ordenada por `Time`.
- **Esquema:** 1842 filas × 31 columnas (mismas columnas que el completo).
- **Ratio de la muestra:** **26.71 %** de fraude — **NO** es el 99:1 original. Es
 una **muestra de laboratorio** pensada para trabajar rápido en local con
 fraudes suficientes. **El desbalance real 99:1 (0.17 %) se trabaja sobre el
 COMPLETO en Colab/caché** (`descargar_datos.py --full` o
 `cargar_fraude_completo`).
- **Peso:** 0.972 MB (≤1 MB, se versiona; coincide con `*muestra*` del `.gitignore`).
- **SHA256 (muestra local):** `2841263a95459c6f0205deda18a8fa1a42e6842e739e8599f998922f9d55e37e`

---

## 3. Manejo del peso y `.gitignore`

El `.gitignore` de la raíz ya cubre esta sesión con la regla general
`Sesiones/*/data/*` + las excepciones whitelisted
(`descargar_datos.py`, `diccionario_datos.md`, `README_datos.md`, `*muestra*`).
Consecuencia verificada:

- `creditcard_muestra.csv` → **se versiona** (coincide con `*muestra*`).
- `creditcard.csv` completo → **bloqueado** (no está en la whitelist) **y** vive
 en el caché temporal fuera de OneDrive → doble seguridad, nunca se comitea.
- `iris.csv` → no versionado (regenerable), pero conservado en disco por liviano.

No se requirió modificar `.gitignore`.

---

## 4. Viabilidad de la réplica y valores OBSERVADOS (no son targets)

> Los **targets cuantitativos con tolerancia** los define `investigador-tema` en
> la ficha de la sesión de réplica del paper (dueño único). Lo de abajo son valores
> **observados** con el venv el 19/07/2026 (smoke `descargar_datos.py --smoke`),
> que confirman que ambos datasets permiten reproducir el resultado; sirven de
> insumo, no de fuente de verdad.

- **LDA sobre Iris** (StratifiedKFold 5, `random_state=42`): accuracy **0.9733**.
- **QDA sobre Iris** (misma CV): accuracy **0.9800**.
 → Ambos dentro del target tentativo declarado para la sesión (0.98 ± 0.03). La réplica de
 Fisher es **viable**; LDA reproduce la separación casi perfecta de las tres
 especies (setosa linealmente separable; solo se confunden algunos versicolor/virginica).
- **Fraude completo (99:1)** — split 70/30 estratificado, `StandardScaler`:
 - Clasificador trivial «todo legítimo»: accuracy **0.9983**, **recall(fraude) = 0.0000**
 → confirma la **paradoja de la accuracy** (target tentativo #3 = 0.00).
 - Línea base `LogisticRegression`: **PR-AUC 0.7080**, ROC-AUC 0.9567,
 recall 0.6149, F1 0.7137 (umbral 0.5).
 - `LDA`: PR-AUC 0.6748 / ROC-AUC 0.9727 / recall 0.7365 / F1 0.7814.
 - `QDA`: PR-AUC 0.1482 / recall 0.8311 / F1 0.1143 (muchos falsos positivos al umbral 0.5).
 - `GaussianNB`: PR-AUC 0.0809 / recall 0.8041 / F1 0.1124.
 - **Efecto SMOTE** (LogReg): recall(fraude) sube de **0.6149 → 0.8784**
 (target tentativo #5: sube post-SMOTE) a costa de precision/F1.

**Nota de contraste ROC vs. PR (relevante para el notebook):** con 0.17 % de
fraude, todos los modelos exhiben ROC-AUC alto (>0.95) mientras el PR-AUC los
separa nítidamente (0.71 LogReg vs. 0.08 GaussianNB). Es exactamente el fenómeno
de Saito & Rehmsmeier (2015): en datos muy desbalanceados el PR es más informativo
que el ROC. El notebook debe reportar **PR-AUC** como métrica de decisión.

---

## 5. Trazabilidad de verificación (19/07/2026)

| Dataset | Fuente usada | Filas × cols | Positivos | SHA256 | Estado |
|---|---|---|---|---|---|
| Iris (repl. LDA) | `sklearn.load_iris` + cruce UCI 53 | 150 × 5 | 50/50/50 | `d32f1072…ca9da9` (local) | Verificado ✓ |
| Credit Card Fraud (negocio) | Mirror HF `David-Egea/Creditcard-fraud-detection` | 284 807 × 31 | 492 (0.1727 %) | `76274b69…551a89` (fuente) | Verificado ✓ (mirror) |
| — muestra de laboratorio | derivada (492 fraudes + 1350 legítimas, rs=42) | 1842 × 31 | 492 (26.71 %) | `2841263a…d55e37e` (local) | Generada ✓ |

**Entorno:** venv 3.13.5; scikit-learn 1.6.1; **imbalanced-learn 0.14.2**
(smoke `from imblearn.over_sampling import SMOTE` OK; SMOTE genera sintéticos sinerror). LDA/QDA (`sklearn.discriminant_analysis`), GaussianNB/MultinomialNB/
BernoulliNB (`sklearn.naive_bayes`) y las métricas `average_precision_score`,
`precision_recall_curve`, `roc_auc_score`, `f1_score`, `log_loss` importan y
corren.
