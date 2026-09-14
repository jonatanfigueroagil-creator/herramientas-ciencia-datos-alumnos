# Diccionario de datos — Sesión S03 (Regresión Lineal Simple — OLS)

> Fecha de verificación de fuentes: **18/07/2026**. Datos descargados y validados
> programáticamente con `descargar_datos.py` (esquema + nº de filas + checksum SHA256).
> Reglas: español impersonal, fechas DD/MM/YYYY, UTF-8. Ningún dato es sintético.
> Ambos CSV pesan < 25 MB → se conservan en la carpeta (no se genera muestra).

---

## 1. `GaltonFamilies.csv` — Dataset de REPLICACIÓN (paper seminal)

- **Rol:** replicación del paper que dio origen al término «regresión».
- **Paper:** Galton, F. (1886). *Regression Towards Mediocrity in Hereditary
 Stature*. **Journal of the Anthropological Institute of Great Britain and
 Ireland 15, 246–263.** Galton observó que los hijos de padres extremos
 (muy altos o muy bajos) tienden a acercarse a la media poblacional:
 «regresión hacia la mediocridad».
- **Procedencia / mirror:** Rdatasets — paquete R `HistData`, dataset
 `GaltonFamilies`, reconstruido del cuaderno original de Galton por
 **Hanley, J. A. (2004)**, *"Transmuting" women into men: Galton's family data
 on human stature*, The American Statistician 58(3), 237–243.
 `https://vincentarelbundock.github.io/Rdatasets/csv/HistData/GaltonFamilies.csv`
- **Licencia:** paquete `HistData` de R bajo licencia **GPL (≥2)**; datos
 históricos de dominio público. Uso académico libre.
- **Tamaño:** **934 filas × 9 columnas**, 34.7 KB (< 25 MB → se conserva),
 **SHA256** `b1fc91e4d4b9604661967dcc62ff1055eaa3020c4d5c42d1f9d588f74b9b3317`.
- **Estructura:** un registro por **hijo/a** (934 hijos de **205 familias**).
 La altura de padre, madre y «mid-parent» se repite dentro de cada familia.
 Todas las alturas están en **pulgadas**.

| Columna | Tipo | Significado (estudio / negocio) |
|---|---|---|
| `rownames` | entero | Índice de fila heredado de R (1–934). No es variable de análisis. |
| `family` | texto | **Identificador de la familia** (001–204 y «136A»); agrupa a los hermanos. 205 familias. |
| `father` | decimal | **Altura del padre** en pulgadas. |
| `mother` | decimal | **Altura de la madre** en pulgadas. |
| `midparentHeight` | decimal | **Altura «mid-parent»**: `(father + 1.08 × mother) / 2`. Galton multiplicó la altura de la madre por 1.08 para ponerla en escala «masculina» antes de promediar. Es el **predictor** de la réplica. |
| `children` | entero | **Nº de hijos** en la familia. |
| `childNum` | entero | **Orden del hijo** dentro de la familia (ordenado por altura decreciente). |
| `gender` | texto | **Sexo del hijo/a**: `male` / `female`. |
| `childHeight` | decimal | **Altura del hijo/a** en pulgadas. Es la **respuesta** (`y`). |

### Réplica — valores OBSERVADOS (smoke test 18/07/2026)
OLS de `childHeight ~ midparentHeight` sobre los 934 hijos. Ejecutado con el venv
(`descargar_datos.py --smoke`), ajustado con **statsmodels** y con **sklearn**:

| Ajuste | Intercepto β₀ | Pendiente β₁ | r | R² | Nota |
|---|---|---|---|---|---|
| Crudo, ambos sexos (`childHeight ~ midparentHeight`) | **22.6362** | **0.6374** | **0.3209** | **0.1030** | Reproduce el clásico ~0.65 |
| Alturas de las hijas × 1.08 (transmutación de Galton/Hanley) | **19.9175** | **0.7126** | **0.4970** | **0.2470** | Sube r y pendiente al homogeneizar sexos |

- **statsmodels y sklearn coinciden exactamente** en el ajuste crudo
 (β₀ = 22.6362, β₁ = 0.6374, R² = 0.1030). Confirma que ambos se importan y se ejecutan.
- **Viabilidad de la réplica: CONFIRMADA.** La pendiente observada sobre
 `midparentHeight` (**0.6374**) **coincide con el target vigente** de la réplica:
 **β₁ = 0.6374 ± 0.03 → [0.6074, 0.6674]**, fijado en la ficha de la sesión de réplica del paper (fila 1 de targets) y espejado en la
 lo declarado para la sesión con su celda `regresion_galton!B2`. El valor reproduce la «regresión hacia
 la media» de Galton porque `midparentHeight` **ya incorpora el factor 1.08** de
 la madre; no hace falta ajustar nada más para obtener ~0.64.
- La **r es baja (0.32)** en el ajuste crudo porque agrupa hijos e hijas sin
 homogeneizar el sexo (la variabilidad entre sexos infla el residual). Al
 multiplicar la altura de las hijas por 1.08, r sube a **0.497** y la pendiente
 a **0.713**. Ambas lecturas son legítimas; el notebook debe **explicitar el
 tratamiento del sexo y de la unidad «mid-parent»** (advertencia de lo declarado para la sesión).
- **Los targets con tolerancia son propiedad de `investigador-tema`**
 (la ficha de la sesión de réplica del paper). Aquí solo se reportan valores OBSERVADOS.

---

## 2. `Advertising.csv` — Dataset de NEGOCIO (laboratorio)

- **Rol:** caso de negocio del laboratorio (OLS simple ventas ~ un canal).
- **Contexto:** inversión publicitaria (en **miles de USD**) en tres canales —
 TV, radio y prensa — y las **ventas** (en **miles de unidades**) de un
 producto en **200 mercados** distintos. Dataset didáctico clásico de ISLR.
- **Fuente / paper:** James, G., Witten, D., Hastie, T. & Tibshirani, R.
 *An Introduction to Statistical Learning* (ISLR), cap. 3. Sitio oficial:
 `https://www.statlearning.com/s/Advertising.csv`.
- **Licencia:** material didáctico de ISLR, distribuido libremente para uso
 educativo desde el sitio oficial de los autores.
- **Tamaño:** **200 filas × 5 columnas**, 4.5 KB (< 25 MB → se conserva),
 **SHA256** `5d88819733db0d929716e9c8fa87f20fe1ec12a9ba2feb43ed7e69e585ace3ad`.
- **Nota de esquema:** el CSV oficial trae una **primera columna índice sin
 nombre** (`Unnamed: 0`, 1–200) además de las 4 variables. Se valida por las 4
 columnas del sílabo; al cargar conviene `index_col=0` para descartar el índice.

| Columna | Tipo | Significado de negocio |
|---|---|---|
| `Unnamed: 0` | entero | Índice del mercado (1–200). No es variable de análisis. |
| `TV` | decimal | **Inversión en publicidad de TV** (miles de USD) en ese mercado. |
| `radio` | decimal | **Inversión en publicidad de radio** (miles de USD). |
| `newspaper` | decimal | **Inversión en publicidad de prensa** (miles de USD). |
| `sales` | decimal | **Ventas** del producto (miles de unidades). Es la respuesta (`y`). |

> Uso en el laboratorio: ajustar un **OLS simple** `sales ~ un canal` (p. ej.> `sales ~ TV`), interpretar β₀/β₁ en unidades de negocio (ventas por cada> S/. o USD invertido), leer R², diagnosticar residuales y medir el error
> fuera de muestra con partición train/test (RMSE / MAE). Regresión **simple**
> con un solo predictor (la múltiple es S04, prohibida aquí).

---

## 3. `anscombe` — Apoyo conceptual (INTEGRADO en seaborn, sin descarga)

- **Rol:** ilustrar por qué **hay que graficar** antes de confiar en los
 estadísticos: cuatro conjuntos con casi idénticas media, varianza,
 correlación y recta OLS, pero formas radicalmente distintas.
- **Paper:** Anscombe, F. J. (1973). *Graphs in Statistical Analysis*.
 **The American Statistician 27(1), 17–21.**
- **Carga:** `seaborn.load_dataset("anscombe")` — **integrado en seaborn**,
 no requiere descarga externa ni archivo local.
- **Verificado (18/07/2026):** carga **44 filas × 3 columnas**
 (`dataset`, `x`, `y`) con **4 datasets** `I`, `II`, `III`, `IV` de
 **11 puntos cada uno**. `descargar_datos.py` incluye `verificar_anscombe`.
- **Licencia:** dataset integrado en seaborn (BSD-3). Uso libre.

| Columna | Tipo | Significado |
|---|---|---|
| `dataset` | texto | Identificador del conjunto del cuarteto: `I`, `II`, `III`, `IV`. |
| `x` | decimal | Variable independiente. |
| `y` | decimal | Variable dependiente. |

---

## 4. Notas de verificación y fallbacks

- **`GaltonFamilies`:** fuente primaria (Rdatasets / `HistData`) respondió 200 y
 validó esquema (9 columnas exactas) y nº de filas (934). Enlace del sílabo
 `medicine.mcgill.ca/.../galton/` está **caído (HTTP-404)**; el fallback
 documentado en el material de referencia de la sesión (GaltonFamilies en Rdatasets)
 se usó como fuente principal y funcionó.
- **`Advertising`:** fuente primaria (`www.statlearning.com/s/Advertising.csv`)
 respondió 200 y validó 4 columnas + 200 filas. El script incluye un **mirror
 ISLR** de respaldo (`raw.githubusercontent.com/nguyen-toan/ISLR/...`); no se
 necesitó ejercitarlo (la primaria funcionó).
- **`anscombe`:** integrado en seaborn 0.13.2; no aplica fallback de descarga.
- **Idempotencia:** `descargar_datos.py` no re-descarga si el archivo ya existe
 y su SHA256 coincide (verificado con dos ejecuciones: la 2.ª no descarga nada).
- **Smoke / viabilidad:** `descargar_datos.py --smoke` reajusta el OLS con
 statsmodels y sklearn y reimprime los valores observados de la tabla Sección 1.
 Confirma que **statsmodels 0.14.6 y scikit-learn 1.6.1 se importan y se ejecutan**.
