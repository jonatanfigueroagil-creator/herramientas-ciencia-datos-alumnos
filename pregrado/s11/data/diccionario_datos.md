# Diccionario de datos — Sesión 11 (Series temporales)

> Verificación real ejecutada con el venv del curso (Python 3.13.5) el **19/07/2026**.
> Todos los datasets se cargaron programáticamente y se validaron filas × columnas × rango.
> Reproducir con `data/descargar_datos.py` (idempotente; local y Colab).

## Resumen

| Rol | Archivo | Filas × Cols | Frecuencia | Rango temporal | Peso | SHA256 |
|---|---|---|---|---|---|---|
| Réplica SARIMA (Box & Jenkins) | `airpassengers.csv` | 144 × 2 | Mensual (MS) | 1949-01 → 1960-12 | 1.9 KB | `ee2d2b1a…4bd4ce` |
| Réplica Prophet (Taylor & Letham) | `peyton_manning.csv` | 2905 × 2 | Diaria (con huecos) | 2007-12-10 → 2016-01-20 | 85 KB | `6b1383a6…5bfa56` |
| Negocio (forecasting demanda) | `train.csv` | 913 000 × 4 | Diaria | 2013-01-01 → 2017-12-31 | 16.5 MB | `038f2569…92f2ed` |
| Negocio — muestra ≤1 MB | `train_muestra.csv` | 18 260 × 4 | Diaria | 2013-01-01 → 2017-12-31 | 340 KB | `c0ea9f25…901916` |

Checksums completos en `data/CHECKSUMS.sha256`.

---

## 1. `airpassengers.csv` — AirPassengers («Serie G»)

**Procedencia.** Box, G. E. P. & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*. Serie internacional mensual de pasajeros de aerolínea; el caso canónico del **modelo aerolínea** SARIMA(0,1,1)(0,1,1)₁₂.
**Fuente operativa.** `statsmodels.datasets.get_rdataset("AirPassengers", "datasets")` (Rdatasets; paquete R `datasets`). El script incluye un **arreglo canónico embebido** como fallback offline (idéntico a la Serie G).
**Licencia.** Dominio público (dataset clásico de dominio académico; Rdatasets GPL-2 sobre el empaquetado).

| Variable | Tipo | Significado de negocio |
|---|---|---|
| `month` | texto `YYYY-MM` | Mes de observación (inicio de mes, frecuencia mensual). |
| `passengers` | entero | Total de pasajeros aéreos internacionales (miles) en el mes. |

Notas: sin faltantes; `passengers` mín 104 (nov-1949) / máx 622 (jul-1960); tendencia creciente y **estacionalidad multiplicativa** (amplitud que crece con el nivel) → se modela sobre `log(passengers)`.

## 2. `peyton_manning.csv` — Wikipedia log page views

**Procedencia.** Taylor, S. J. & Letham, B. (2018). *Forecasting at Scale*. The American Statistician 72(1). Dataset de ejemplo oficial de **Prophet**: logaritmo de las visitas diarias a la página de Wikipedia de Peyton Manning.
**Fuente operativa.** Descarga directa del repositorio oficial de Prophet:
`https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv`
**Licencia.** MIT (repositorio `facebook/prophet`).

| Variable | Tipo | Significado de negocio |
|---|---|---|
| `ds` | fecha `YYYY-MM-DD` | Día de la observación (serie diaria; **contiene huecos** — no todos los días figuran). |
| `y` | flotante | log(visitas diarias). Rango observado 5.263 – 12.847. |

Notas: 2905 observaciones entre 2007-12-10 y 2016-01-20 (≈8 años); fuerte **estacionalidad semanal y anual** y **picos** ligados a partidos/playoffs (útiles para ilustrar changepoints y holidays de Prophet). Serie con calendario irregular: NO asumir frecuencia diaria estricta.

## 3. `train.csv` — Store Item Demand Forecasting Challenge

**Procedencia.** Competencia de Kaggle `c/demand-forecasting-kernels-only` (*Store Item Demand Forecasting Challenge*). El caso de **negocio** (forecasting de demanda multi-serie con backtesting de origen móvil), no la fuente de una réplica de paper.
**Acceso.** El enlace original de Kaggle **exige login** (no API sin credenciales). Se resolvió con un **mirror abierto byte-idéntico**:
- Primario: `https://raw.githubusercontent.com/jgonzalezab/Store-Item-Demand-Forecasting/master/Data/train.csv`
- Secundario: `https://raw.githubusercontent.com/ugursaricam/store-item-demand-forecasting/master/datasets/train.csv`

Ambos mirrors devuelven exactamente los mismos 17 333 449 bytes y el mismo SHA256 (`038f2569…92f2ed`), verificado el 19/07/2026. La descarga manual desde Kaggle se documenta en `data/README_datos.md`.
**Licencia.** Reglas de la competencia de Kaggle (uso académico/educativo). El mirror se emplea solo como vía de acceso; la fuente citada es Kaggle.

| Variable | Tipo | Significado de negocio |
|---|---|---|
| `date` | fecha `YYYY-MM-DD` | Día de la venta (serie diaria completa, sin huecos). |
| `store` | entero 1–10 | Identificador de tienda (10 tiendas). |
| `item` | entero 1–50 | Identificador de producto (50 productos). |
| `sales` | entero ≥0 | Unidades vendidas del producto en la tienda ese día. |

Estructura: **500 series** (10 tiendas × 50 items) × **1826 días** (2013-01-01 → 2017-12-31) = 913 000 filas; sin faltantes. `sales` mín 0 / máx 231 / media 52.25.
**Diseño de la sesión.** El entregable pide forecasting **mensual** de la demanda de **un producto** → agregar `sales` por (`store`,`item`,mes) genera 60 puntos mensuales por serie, con tendencia y estacionalidad anual, aptos para SARIMA/ETS/Prophet y backtesting rolling-origin. Ideal también para features de lag/rolling/calendario en el forecasting con ML.

### `train_muestra.csv` (muestra ≤1 MB)
Subconjunto de `train.csv`: **tienda 1, items 1–10** (18 260 filas, 340 KB), con las 10 series diarias completas 2013–2017. Mismo esquema que `train.csv`. Permite ejecutar el notebook sin descargar los 16.5 MB (p. ej. en Colab con poca red). El archivo completo se conserva en `data/` por estar por debajo del umbral de 25 MB.

---

## Notas de viabilidad de la réplica (no son targets — dueño: la ficha de la sesión de réplica del paper)

Ejecutado con el venv el 19/07/2026 como control de viabilidad (NO define targets; solo confirma que los datos permiten reproducir el paper):

- **SARIMA aerolínea** `SARIMAX((0,1,1),(0,1,1),12)` sobre `log(passengers)` ajusta sin error: AIC ≈ **−435.44**, θ₁ (`ma.L1`) ≈ **−0.433**, Θ₁ (`ma.S.L12`) ≈ **−0.548**. Coeficientes MA negativos, coherentes con el modelo aerolínea clásico. *(Los valores exactos y sus tolerancias los fija `REPLICACION_PAPER.md`; statsmodels puede diferir levemente de las cifras publicadas en R/`forecast`.)*
- **Estacionariedad** de `log(passengers)`: ADF p ≈ 0.42 (no rechaza raíz unitaria) y KPSS p ≤ 0.01 (rechaza estacionariedad) → confirma la necesidad de diferenciación (d=1, D=1), como en la sección 2 declarado para la sesión.
- **STL** descompone la serie sin error (period=12).
- **Prophet** ajusta sobre Peyton Manning (2905 filas) y genera pronóstico + intervalos sin error; `cross_validation`/`performance_metrics` disponibles para el MAPE de backtesting.

**Viabilidad: los tres datasets reproducen el diseño de la sesión sin obstáculos.**
