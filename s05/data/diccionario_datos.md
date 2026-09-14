# Diccionario de datos — S05 (Regularización + regresión no lineal)

> **Fecha de verificación de fuentes: 18/07/2026.** Descarga y validación reales con el venv del curso (`descargar_datos.py --smoke`). PROHIBIDO inventar datos: todas las cifras de este documento se OBSERVARON al cargar los archivos.

Dos datasets, dos roles:

| Archivo | Rol | Filas × Cols | Fuente (verificada 18/07/2026) | SHA256 del archivo guardado |
|---|---|---|---|---|
| `prostate.csv` | **Replicación del paper** (ancla ESL Tabla 3.3 / Tibshirani 1996) | **97 × 11** | Mirror GitHub `empathy87/The-Elements-of-Statistical-Learning-Python-Notebooks`, archivo `data/Prostate Cancer.txt` (el enlace del sílabo `hastie.su.domains/.../prostate.data` da HTTP-403 a scripts). | `73c2a7c8b05beb72e202b17884fea53d0fe0db1e25807934bfe824198f4a271b` |
| `communities.csv` | **Negocio / alta dimensión** (~100 predictores) | **1994 × 128** | UCI ML Repository (id 183), `communities.data` + `communities.names`. Fallback: `fetch_openml(name="us_crime")`. | `f6d3ff2ca832aafc0d1b7f935d6f39647bd3f3f17436c60ff3009ccf99352460` |

- SHA256 de los **bytes crudos de la fuente**: prostate `009f1732…1671f6` — communities `09e0b5c0…24da2ea`.
- Ambos archivos pesan < 1.2 MB → se conservan en OneDrive (la regla de exclusión aplica solo a >25 MB).
- Kaggle no interviene en esta sesión; ambas fuentes son abiertas y el alumno nunca necesita API key.

---

## 1. PROSTATE — replicación (Stamey et al., 1989 / ESL cap. 3)

### Qué es y por qué se usa
Estudio de **97 hombres a punto de recibir una prostatectomía radical**. Se mide el nivel de **antígeno prostático específico (PSA)** y varias medidas clínicas; el objetivo es predecir el **log del PSA (`lpsa`)** a partir de ellas. Es el conjunto canónico con el que *The Elements of Statistical Learning* (Hastie, Tibshirani, Friedman) ilustra en el capítulo 3 (Tabla 3.3) la comparación OLS / Best-subset / **Ridge / Lasso** / PCR / PLS, y el que usa Tibshirani (1996) para presentar el LASSO. Trae una **partición train/test fija** (columna `train`) que permite reproducir sus números exactos.

**Citas:**
- Stamey, T.A., Kabalin, J.N., McNeal, J.E., et al. (1989). *Prostate specific antigen in the diagnosis and treatment of adenocarcinoma of the prostate. II. Radical prostatectomy treated patients.* The Journal of Urology 141(5), 1076-1083. ← paper de origen del dato.
- Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso.* JRSS-B 58(1), 267-288. ← paper de la técnica (réplica).
- Hastie, Tibshirani & Friedman (2009). *The Elements of Statistical Learning*, cap. 3, Tabla 3.3.

### Procedencia y licencia
Datos publicados por Stamey et al. y redistribuidos con el libro ESL (paquete R `ElemStatLearn`). Uso libre para docencia e investigación; sin datos personales identificables (97 pacientes anonimizados).

### Variables (11 columnas)
> Todos los predictores son numéricos. `lpsa` es la variable objetivo. Varias variables ya vienen en escala logarítmica (prefijo `l`).

| Columna | Significado (negocio/clínico) | Tipo |
|---|---|---|
| `id` | Índice 1–97 del registro (lo añade el mirror; **no es predictor**). | entero |
| `lcavol` | Log del **volumen del cáncer** (log cancer volume). **El predictor más fuerte de `lpsa`.** | float |
| `lweight` | Log del **peso de la próstata** (log prostate weight). | float |
| `age` | **Edad** del paciente (años). | entero |
| `lbph` | Log de la **hiperplasia prostática benigna** (benign prostatic hyperplasia). | float |
| `svi` | **Invasión de vesículas seminales** (seminal vesicle invasion), binaria 0/1. | binaria |
| `lcp` | Log de la **penetración capsular** (capsular penetration). | float |
| `gleason` | **Puntaje de Gleason** (agresividad del tumor, escala 6–9). | ordinal |
| `pgg45` | **Porcentaje de puntajes Gleason 4 o 5** históricos. | float (%) |
| `lpsa` | **VARIABLE OBJETIVO**: log del antígeno prostático específico (log PSA). | float |
| `train` | Partición de ESL: `T` = entrenamiento (**67**), `F` = prueba (**30**). | T/F |

### ⚠️ Nota de orden del archivo (medido el 15/08/2026): las filas vienen ordenadas por la RESPUESTA
Las 97 filas de `prostate.csv` están ordenadas de forma **estrictamente creciente por `lpsa`** (de −0,431 a 5,583; `corr(nº de fila, lpsa) = 0,9581`; `id` es simplemente ese orden). Dos consecuencias operativas:

1. **No afecta a la partición train/test**, porque la columna `train` viene **entremezclada** (`T`/`F` alternados, no bloques): T = 67 y F = 30 cubren todo el rango de `lpsa`. La partición de ESL se usa **tal cual**, sin re-barajar.
2. **Sí afecta a la validación cruzada.** `KFold` **no baraja por defecto**: con `cv=10` (un entero) en `LassoCV`/`ElasticNetCV`/`validation_curve`, cada pliegue sería una **franja contigua** de `lpsa` y el λ elegido pasaría a ser un artefacto del orden del archivo. Medido sobre el train de 67 filas: CV-MSE mínimo **0,7565** sin barajar frente a **0,5359** con `KFold(n_splits=10, shuffle=True, random_state=42)`, y el Elastic Net cambia de mezcla (α 0,1 → 0,5). **El notebook y el validador barajan siempre.** Detalle en la guía de supuestos de la sesión «Sección 2.5» y en la hoja `cv_barajado` de `resultados/S05_resultados.xlsx`.

*El archivo **no** se reordena:* alterarlo rompería la comparabilidad con ESL y con el mirror verificado por checksum. El orden se **documenta** y se corrige en el método (barajar los pliegues), no en el dato.

**Nota de lectura en clase:** por el mismo motivo, un `head(3)` de este CSV muestra **los tres pacientes con menor `lpsa`** del estudio, no una muestra representativa (mediana de `lcavol` = 1,447 frente a −0,580 / −0,994 / −0,511 en esas tres filas).

### Nota de réplica: convención de estandarización (IMPORTANTE, valores OBSERVADOS)
Los predictores **deben estandarizarse a varianza unitaria** para Ridge/Lasso. **La convención de ESL estandariza sobre las 97 observaciones** (no solo sobre las 67 de train). Con esa convención el smoke reproduce ESL Tabla 3.3 **exactamente**:

| Cantidad | Observado (venv, 18/07/2026) | ESL Tabla 3.3 |
|---|---|---|
| Coef. OLS de `lcavol` (train estandarizado) | **0.6760** | 0.68 |
| Intercepto OLS | **2.4649** | 2.46 |
| **Error de test (MSE) del modelo OLS** | **0.5213** | **0.521** |

> **Nota de discrepancia con el target tentativo declarado para la sesión (Sección 8, «ESL ~0.68»):** el valor depende de la *scope* de la estandarización.
> - Al estandarizar sobre las **97** (convención ESL) → `lcavol` = **0.676** (≈0.68 ✓).
> - Al estandarizar **solo sobre el train (67)** → `lcavol` = **0.711**; el **error de test todavía es 0.521** (OLS es invariante a re-escalados afines aplicados por igual a train y test; solo cambia la *escala* del coeficiente reportado).
>
> **Recomendación para el notebook y para la ficha de la sesión de réplica del paper (dueño del target):** para reproducir la *tabla de coeficientes* de ESL, estandarizar sobre las 97; para una práctica ML "pura" (sin fuga de información) se estandariza sobre el train y se reporta el error de test (0.521, idéntico). Conviene explicitar esta convención en clase.

### Otros valores OBSERVADOS de viabilidad (smoke, no son targets)
- **LASSO** (estandarización ESL): al crecer λ anula variables en este orden → `gleason` (α=0.01), luego `lcp`, `age`, `lbph`, `pgg45`. Ejemplos: α=0.05 → 6/8 distintas de cero; α=0.1 → 5/8; α=0.3 → 3/8. Confirma el drill 1 declarado para la sesión («LASSO lleva coeficientes exactamente a cero»).
- **Ridge** (α=1.0): coef(`lcavol`)=0.6535, MSE_test=0.5127.
- **ElasticNetCV** (l1_ratio∈{.1,.5,.7,.9,.95,1}): elige l1_ratio\*=1.0 (LASSO puro), α\*≈0.0009, 8/8 variables, MSE_test=0.5186.

---

## 2. COMMUNITIES AND CRIME — negocio / alta dimensión (Redmond, 2009)

### Qué es y por qué se usa
**1994 comunidades de EE. UU.** descritas por variables socioeconómicas (censo de EE. UU. de 1990) y policiales (encuesta LEMAS de 1990), con la tasa de crímenes violentos (UCR 1995 del FBI) como objetivo. **Todas las variables numéricas están normalizadas al rango [0, 1]** (normalización decimal por el creador del dataset). Es el caso de **alta dimensión** del laboratorio: ~100 predictores, muchos correlacionados y con faltantes → escenario donde OLS falla y **Ridge/LASSO/Elastic Net** destacan (selección de variables y control de varianza). Es el dataset del **entregable evaluable** de la sesión.

**Cita:** Redmond, M. (2009). *Communities and Crime Data Set.* UCI Machine Learning Repository. DOI: 10.24432/C53W3X. Combina censo 1990 de EE. UU., encuesta LEMAS 1990 y Uniform Crime Report 1995 del FBI.

### Procedencia y licencia
UCI Machine Learning Repository (dataset id 183). Uso libre para investigación y docencia con atribución (licencia CC BY 4.0 del repositorio UCI). Datos agregados a nivel de comunidad; sin registros individuales.

### Estructura de las 128 columnas
- **5 columnas NO predictivas** (identificadores/control): `state` (código de estado; 46 estados presentes), `county`, `community`, `communityname` (nombre de la comunidad), `fold` (pliegue 1–10 para CV, definido por el creador). El paper recomienda **no** usar estas 5 como predictores.
- **122 predictores** normalizados a [0,1] (conteo en crudo; el **modelado descarta el bloque LEMAS** y trabaja con **100 predictores** — ver «Valores OBSERVADOS de viabilidad» abajo). Bloques temáticos principales:
 - **Demografía y raza**: `population`, `householdsize`, `racepctblack`, `racePctWhite`, `racePctAsian`, `racePctHisp`, franjas etarias (`agePct12t21`, …).
 - **Ingreso y pobreza**: `medIncome`, `medFamInc`, `perCapInc`, `pctWWage`, `PctPopUnderPov`, `NumUnderPov`, `whitePerCap`/`blackPerCap`/…
 - **Educación y empleo**: `PctLess9thGrade`, `PctNotHSGrad`, `PctBSorMore`, `PctUnemployed`, `PctEmploy`, `PctEmplManu`, `PctOccupMgmtProf`.
 - **Estructura familiar**: `MalePctDivorce`, `TotalPctDiv`, `PctFam2Par`, `PctKids2Par`, `PctIlleg`, `PctWorkMom`.
 - **Inmigración e idioma**: `PctImmigRec5/8/10`, `PctRecentImmig`, `PctNotSpeakEnglWell`, `PctForeignBorn`.
 - **Vivienda**: `PctHousOwnOcc`, `MedRent`, `MedOwnCostPctInc`, `HousVacant`, `PctVacantBoarded`, `MedYrHousBuilt`, `OwnOccMedVal`.
 - **Policía (bloque LEMAS)**: `LemasSwornFT`, `PolicPerPop`, `PolicOperBudg`, `RacialMatchCommPol`, `PctPolic*`, `PolicBudgPerPop`, … → **este bloque concentra los faltantes (ver abajo)**.
- **1 variable objetivo**: **`ViolentCrimesPerPop`** — crímenes violentos por 100 000 habitantes, normalizada a [0,1] (rango observado 0.0–1.0; 0 faltantes).

### Manejo de faltantes «?» (valores OBSERVADOS)
En `communities.data` los faltantes se codifican con **`?`**; `descargar_datos.py` los convierte a `NaN` al guardar el CSV. Observado (18/07/2026):
- **25 columnas** con al menos un faltante (de 128).
- **22 columnas del bloque policial/LEMAS** tienen exactamente **1675 faltantes = 84.0 %** cada una (p. ej. `LemasSwornFT`, `PolicPerPop`, `RacialMatchCommPol`, `PolicOperBudg`, `PolicBudgPerPop`): solo ~319 comunidades respondieron la encuesta LEMAS. **Práctica recomendada del laboratorio: descartar ese bloque** en lugar de imputar (imputar el 84 % introduce sesgo).
- `OtherPerCap` tiene **1** faltante (imputable trivialmente).
- Los identificadores `county` y `community` están mayormente ausentes (no son predictores → se descartan).
- **La variable objetivo `ViolentCrimesPerPop` no tiene faltantes** (1994 valores válidos).

### Valores OBSERVADOS de viabilidad (smoke, no son targets)
Tras descartar las 5 columnas no predictivas y el objetivo (→ **122 predictores en crudo**) y **descartar el bloque policial LEMAS** (22 columnas con ~84 % faltante; imputar el 84 % sesga) quedan **100 predictores modelados**; se imputan los faltantes residuales (p. ej. `OtherPerCap`) por la media, se estandariza y se parte 70/30 con semilla fija:
- **OLS** (100 predictores): MSE_test = 0.01753.
- **RIDGE** (`RidgeCV`): MSE_test = 0.01744, mantiene las 100 (contrae hacia cero, no selecciona).
- **LASSO por CV** (`LassoCV`): deja **62 de 100** variables distintas de cero, MSE_test = 0.01747.
- **Elastic Net** (`ElasticNetCV`): deja **72 de 100** variables distintas de cero, MSE_test = 0.01747.

Mensaje central del entregable: el **LASSO iguala el error del OLS (0.01753) con 62 de 100 predictores (deja fuera 38)** y el Elastic Net 72; el **RIDGE mantiene las 100** (contrae hacia cero, no selecciona) → **interpretabilidad vs. desempeño**.

---

## 3. Reproducir la descarga

```bash
# local (venv del curso)
"C:/Users/Usuario/OneDrive/Cursos/Herramientas de Ciencias de Datos/python" descargar_datos.py --smoke
# Colab
%run descargar_datos.py
```

El script es idempotente (si el CSV existe con el esquema correcto, no re-descarga), verifica filas × columnas y columnas clave (incluida la partición 67/30 de `prostate` y el objetivo de `communities`), y funciona en local y en Colab.

## 4. Archivos hermanos (referencias cruzadas)
- Generador: `data/descargar_datos.py`.
- Réplica y targets con tolerancia (dueño único): la ficha de la sesión de réplica del paper.
- Interpretación de negocio: la guía de interpretación de resultados.
- Contrato de la sesión: el cuaderno de la sesión.
- Metadatos originales UCI: `data/communities.names`.
