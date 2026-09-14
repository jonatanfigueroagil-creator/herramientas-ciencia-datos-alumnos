# Diccionario de datos — Ames Housing + Insurance (S04, Regresión Lineal Múltiple)

> **Fecha de verificación de fuentes: 18/07/2026 (Ames) y 14/09/2026 (Insurance).** Descarga y validación reales con el venv del curso (`descargar_datos.py`). PROHIBIDO inventar datos: todas las cifras de este documento se OBSERVARON al cargar los archivos.
>
> **Regla del usuario (13-14/09/2026):** el notebook resuelve un solo dataset completo (el de réplica del paper, `AmesHousing.csv`); el dataset de negocio del entregable debe ser **distinto** al que el notebook ya resuelve. `AmesHousing_kaggle.csv` es la misma base de datos que `AmesHousing.csv` en otra muestra —no un caso de negocio genuino—, así que el entregable pasa a usar **`insurance.csv`** (Sección 3b), un dataset de una aseguradora de salud sin relación con Ames. `AmesHousing_kaggle.csv` queda documentado (Sección 3) pero ya no es el dataset del entregable.

## 1. Qué es Ames Housing y por qué se usa

Registro de **ventas residenciales en la ciudad de Ames, Iowa (EE. UU.), entre 2006 y 2010**, con precio de venta y ~80 características físicas y de calidad de cada vivienda. Dean De Cock lo publicó en 2011 como **alternativa docente al Boston Housing** (este último retirado de scikit-learn por una variable con sesgo racial). Es el conjunto canónico para enseñar regresión múltiple: mezcla variables numéricas, ordinales y categóricas, tiene multicolinealidad real (p. ej. `Garage Cars` vs `Garage Area`), heterocedasticidad y unos pocos atípicos bien documentados.

**Cita del paper (réplica):** De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project.* Journal of Statistics Education, 19(3). DOI: 10.1080/10691898.2011.11889627.

## 2. Procedencia y licencia

- **Origen primario de los datos:** Ames City Assessor's Office (registro público de tasaciones/ventas).
- **Publicación académica:** el propio autor aloja el archivo en el Journal of Statistics Education. **Uso libre para docencia e investigación** (dataset de dominio educativo, redistribuible con atribución al paper).
- **Anonimizado:** identifica cada vivienda por `PID` (parcel ID) y `Order`; sin datos personales de vendedores o compradores.

## 3. Los dos archivos de esta sesión

| Archivo | Rol | Filas × Cols | Fuente (verificada 18/07/2026) | SHA256 del archivo guardado |
|---|---|---|---|---|
| `AmesHousing.csv` | **Replicación del paper** (ancla los números de De Cock) | **2930 × 82** | Journal of Statistics Education (hosting del autor): `http://jse.amstat.org/v19n3/decock/AmesHousing.txt` (tab-sep). Fallback: Rdatasets `openintro/ames`. | `5d5d25c60c165143749013a89a86c604fd55bac07c2a2e6e0d6c9dacf65e6226` |
| `AmesHousing_kaggle.csv` | Muestra alterna de la misma base (**ya no es el dataset de negocio** — ver nota arriba y Sección 3b) | **1460 × 81** | `sklearn.datasets.fetch_openml(data_id=42165)` (OpenML espeja el *train* de la competencia Kaggle *House Prices*). | `25273b2e062575f43bad23211f4f9d0df28502c95cd8cc84a4172bfe552b6e93` |

- **SHA256 de la fuente primaria (bytes crudos del.txt de JSE):** `6cfe6cb525ba437de428653a1040e2aed7d696640bf75203786a6d7a0e67cfcc`.
- El subconjunto Kaggle (1460) es un **~50 % del conjunto completo**; por eso reproduce la correlación pero **no** las 5 ventas atípicas exactas (solo contiene 4 de las 5). Para replicar a De Cock se usa SIEMPRE `AmesHousing.csv` (2930).
- Ambos archivos pesan < 1.5 MB → se conservan en OneDrive (la regla de exclusión aplica solo a >25 MB).
- **Kaggle exige login** (decisión del curso 17/07/2026: sin token de API). Por eso el subconjunto se carga vía OpenML y el conjunto completo vía JSE; el alumno nunca necesita una API key.

### Nota sobre nombres de columna (importante)

Las dos versiones nombran las mismas variables de forma distinta:

| Concepto | `AmesHousing.csv` (De Cock, canónico) | `AmesHousing_kaggle.csv` (Kaggle) |
|---|---|---|
| Precio de venta | `SalePrice` | `SalePrice` |
| Área habitable sobre rasante | `Gr Liv Area` (con espacios) | `GrLivArea` (sin espacios) |
| Calidad general | `Overall Qual` | `OverallQual` |
| Año de construcción | `Year Built` | `YearBuilt` |
| Barrio | `Neighborhood` | `Neighborhood` |

El notebook usa `AmesHousing.csv` para la réplica (nombres con espacios) y puede usar cualquiera de los dos para el laboratorio; conviene documentar en clase esta diferencia de convención.

## 3b. `insurance.csv` — dataset de negocio del entregable (nuevo, 14/09/2026)

- **Qué es:** registro de **1338 asegurados** de una aseguradora de salud en EE. UU., con la prima anual facturada (`charges`) y seis variables demográficas/de comportamiento. Publicado originalmente por Brett Lantz en *Machine Learning with R* (Packt, 2013).
- **Rol:** dataset de negocio del entregable — el notebook no lo usa; el alumno lo trabaja por primera vez en el entregable, aplicando ahí las técnicas de regresión múltiple, dummies, interacción y diagnóstico de supuestos que el notebook enseñó sobre `AmesHousing.csv`.
- **Procedencia y licencia:** **dominio público**, declarado por el propio autor («all of these datasets are in the public domain but simply needed some cleaning up and recoding», Lantz 2013). Mirror verificado: `https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv` (también distribuido como «Medical Cost Personal Datasets» en Kaggle).
- **Dimensiones verificadas (14/09/2026):** 1338 filas × 7 columnas, sin valores faltantes.
- **SHA256 (archivo local):** `505c1cbc2e63d0363bac59501563df2530aadf4cdb9cfee226f4ef32f5468281`.
- **Caso de negocio:** aseguradora de salud privada que necesita tarifar el riesgo de un asegurado y priorizar la revisión de pólizas mal tarifadas — encaja con la regla del curso de casos de negocio en industria privada (aseguradora), nunca gobierno/municipalidad.

| Columna | Tipo | Significado de negocio |
|---|---|---|
| `age` | entero | Edad del asegurado (18–64 años). |
| `sex` | categórica (2) | Sexo (`female`/`male`). Categoría de referencia recomendada: `female`. |
| `bmi` | decimal | Índice de masa corporal (kg/m²). Eje de la interacción con `smoker`. |
| `children` | entero | Número de hijos/dependientes cubiertos por la póliza (0–5). |
| `smoker` | categórica (2) | Condición de fumador (`yes`/`no`). **El predictor individual más fuerte de `charges`**; categoría de referencia recomendada: `no`. |
| `region` | categórica (4) | Región de residencia en EE. UU. (`northeast`, `northwest`, `southeast`, `southwest`). Categoría de referencia recomendada: `northeast`. |
| `charges` | decimal | **Variable objetivo.** Prima médica anual facturada, en USD (326–63 770; mediana 9 382, media 13 270). Fuertemente asimétrica a la derecha — candidata directa a `log(charges)`. |

**Valores observados de viabilidad (venv del curso, 14/09/2026)**, `log(charges) ~ age + bmi + children + sex + smoker + region + smoker:bmi`:
- R² = 0.7835 (ajustado 0.7820); sin la interacción, R² = 0.7679 (ajustado 0.7666).
- VIF: todos los predictores < 1.7 (sin multicolinealidad severa — contraste útil con `communities.csv` de S05, donde sí la hay).
- Breusch-Pagan: LM ≈ 70.49, p < 0.001 (rechaza homocedasticidad; corrección = errores robustos HC3).
- Durbin-Watson ≈ 2.036 (sin autocorrelación — corte transversal sin orden significativo).
- CV 5-fold (RMSE en USD, retransformado con smearing de Duan): sin interacción 9571 ± 960; con interacción 10387 ± 1325 (el R² in-sample sube con la interacción, pero el RMSE de CV no mejora — ver nota de honestidad del entregable).
- Cook's D > 4/n: 93 de 1338 observaciones (mayormente fumadores de BMI alto).
- Coeficiente de `smoker_yes` (modelo sin interacción, log): β ≈ 1.5543 → exacto $100(e^{1.5543}-1)\approx 373\,\%$ (≈4.7× la prima, *ceteris paribus*).

## 4. Variables clave (significado de negocio) — Ames Housing (réplica del notebook)

> Nombres canónicos de De Cock (`AmesHousing.csv`). Entre paréntesis, el nombre Kaggle si difiere.

### Variable objetivo
- **`SalePrice`** — Precio de venta de la vivienda, en dólares estadounidenses (USD). Es lo que el modelo predice. Rango observado ~$12 789 – $755 000; distribución sesgada a la derecha (candidata a transformación log/Box-Cox, contenido de la sesión).

### Tamaño y superficie (numéricas continuas)
- **`Gr Liv Area`** (`GrLivArea`) — Área habitable sobre el nivel del suelo, en pies cuadrados (sq ft). **El predictor individual más importante del precio**; el ejemplo canónico de De Cock es `SalePrice ~ Gr Liv Area`.
- **`Total Bsmt SF`** (`TotalBsmtSF`) — Superficie total del sótano (sq ft). Aporta valor pero no siempre es "habitable".
- **`Lot Area`** (`LotArea`) — Superficie del lote/terreno (sq ft).
- **`1st Flr SF` / `2nd Flr SF`** — Superficie del primer y segundo piso (sq ft).
- **`Garage Area`** (`GarageArea`) — Superficie del garaje (sq ft). **Muy correlacionada con `Garage Cars`** → ejemplo de multicolinealidad (VIF observado ≈ 5).

### Calidad y estado (ordinales / discretas)
- **`Overall Qual`** (`OverallQual`) — Calidad general de materiales y acabados, escala **1 (muy pobre) a 10 (excelente)**. Segundo predictor más fuerte; se trata como numérica ordinal.
- **`Overall Cond`** (`OverallCond`) — Estado general de conservación, escala 1–10.
- **`Garage Cars`** (`GarageCars`) — Capacidad del garaje en número de autos (0–4). Colineal con `Garage Area`.
- **`Full Bath`** (`FullBath`) — Baños completos sobre rasante.
- **`Bedroom AbvGr`** (`BedroomAbvGr`) — Dormitorios sobre rasante.
- **`TotRms AbvGrd`** (`TotRmsAbvGrd`) — Total de habitaciones sobre rasante (sin contar baños).

### Antigüedad (numéricas discretas)
- **`Year Built`** (`YearBuilt`) — Año de construcción original. Proxy de antigüedad; a más reciente, mayor precio en promedio.
- **`Year Remod/Add`** (`YearRemodAdd`) — Año de la última remodelación/ampliación (= año de construcción si no hubo remodelación).
- **`Yr Sold`** (`YrSold`) / **`Mo Sold`** (`MoSold`) — Año y mes de la venta (2006–2010); útiles para efectos de mercado/estacionalidad.

### Categóricas (para dummies / one-hot e interacciones)
- **`Neighborhood`** — Barrio dentro de Ames (28 categorías: NridgHt, StoneBr, OldTown, etc.). **Eje de la interacción "zona × tamaño"** del laboratorio: el valor de un pie cuadrado adicional depende del barrio.
- **`MS Zoning`** (`MSZoning`) — Clasificación de zonificación (residencial de baja/alta densidad, comercial, etc.).
- **`Bldg Type`** (`BldgType`) — Tipo de edificación (unifamiliar, dúplex, adosada…).
- **`House Style`** (`HouseStyle`) — Estilo/número de pisos (1Story, 2Story, etc.).
- **`Central Air`** (`CentralAir`) — Aire acondicionado central (Y/N); dummy binaria directa.
- **`Sale Condition`** (`SaleCondition`) — Condición de la venta (**Normal**, Abnorml, Partial, Family, Alloca, AdjLand). **Clave para la limpieza de atípicos** (ver Sección 5).

### Identificadores (no predictores)
- **`Order`** — Índice secuencial del registro (1–2930). Solo en el archivo completo.
- **`PID`** — Parcel ID del catastro. Solo en el archivo completo.
- **`Id`** — Identificador de fila en el archivo Kaggle. No usar como predictor.

## 5. Nota de limpieza: las 5 ventas atípicas de De Cock (verificado)

De Cock recomienda **remover las viviendas con `Gr Liv Area` > 4000 sq ft** antes de modelar. En `AmesHousing.csv` (2930) son **exactamente 5** casas (valor OBSERVADO 18/07/2026):

| Gr Liv Area | SalePrice | Sale Condition | Interpretación |
|---:|---:|---|---|
| 5642 | 160 000 | Partial | Venta no de mercado (obra sin terminar) — atípico "malo" |
| 5095 | 183 850 | Partial | Venta no de mercado — atípico "malo" |
| 4676 | 184 750 | Partial | Venta no de mercado — atípico "malo" |
| 4476 | 745 000 | Abnorml | Mansión legítima cara — atípico "bueno" |
| 4316 | 755 000 | Normal | Mansión legítima cara — atípico "bueno" |

Tres son **ventas parciales** (precio no representa el mercado) y dos son **casas de gran superficie vendidas a precio coherente**. Removerlas las cinco (criterio `Gr Liv Area <= 4000`) es la práctica que fija De Cock y que el laboratorio replica. Efecto observado: la correlación `SalePrice ~ Gr Liv Area` sube de **0.7068** (todas) a **0.7195** (sin las 5).

## 6. Valores observados de viabilidad (smoke, 18/07/2026)

Sobre `AmesHousing.csv` (2930), tras remover las 5 atípicas (n = 2925), con el venv del curso:

- **Correlación** `SalePrice ~ Gr Liv Area`: 0.7068 (todas) → 0.7195 (limpio).
- **OLS múltiple** `SalePrice ~ Gr Liv Area + Overall Qual` (statsmodels): R² = 0.7513, R²aj = 0.7511; coeficientes ≈ 62.8 USD/sq ft y 32 797 USD por punto de calidad; todos los p-value < 0.001.
- **Errores robustos** HC3 (White) para `Gr Liv Area`: SE 2.33 vs 1.81 clásico → indicio de heterocedasticidad (contenido de la sesión).
- **VIF** `Gr Liv Area` 1.50, `Overall Qual` 2.20, `Year Built` 1.59 (bajos); `Garage Cars` 5.05 y `Garage Area` 4.94 (colinealidad demostrable).
- **CV 5-fold** (sklearn, `SalePrice ~ Gr Liv Area + Overall Qual + Year Built`): R² medio 0.7740 (± 0.0163).

> Estos son valores OBSERVADOS de viabilidad, no los *targets* de la réplica. Los targets con tolerancia los fija la ficha de la sesión de réplica del paper (dueño único: `investigador-tema`).

## 7. Reproducir la descarga

```bash
# local (venv del curso)
"d:/OneDrive/Cursos/Herramientas de Ciencias de Datos/python" descargar_datos.py --smoke
# Colab
%run descargar_datos.py
```

El script es idempotente (si el CSV existe con el esquema correcto, no re-descarga), verifica filas × columnas y columnas clave, y funciona en local y en Colab.

## 8. Archivos hermanos (referencias cruzadas)

- Generador: `data/descargar_datos.py`.
- Resumen para el estudiante: `plantillas/diccionario_ames.md` (subconjunto de este diccionario).
- Réplica y targets: la ficha de la sesión de réplica del paper.
- Contrato de la sesión: el cuaderno de la sesión.
