# Diccionario de datos — E7 (EPE) Sistemas de recomendación y proyecto integrador

> Fecha de verificación: **06/08/2026**; verificación repetida el 07/08/2026 con `data/descargar_datos.py` sobre el venv del curso (la matriz de versiones certificada del curso). Verificación REAL (carga programática y validación de conteos), no solo HTTP 200.

Dos datasets: uno de **películas** (MovieLens 100k, feedback explícito 1-5) para el recomendador colaborativo, y uno de **negocio** (Online Retail II, feedback implícito) para el recomendador de e-commerce. En EPE se usan a nivel de **intuición y decisión** (sin réplica formal de paper en esta sesión).

---

## 1. MovieLens 100k — recomendador de películas (feedback explícito)

**Rol:** base del recomendador colaborativo (vecindario item-based + factorización SVD) y del filtrado por contenido (géneros).

**Base original: SUSTITUTO canónico.** El dataset sobre el que Koren, Bell & Volinsky (2009) desarrollaron la factorización fue el **Netflix Prize**, **retirado de distribución** (litigio de privacidad, 2010). **MovieLens 100k** (GroupLens) es el sustituto canónico con el que la literatura reproduce la factorización sobre rating explícito 1-5. En EPE se menciona sin réplica formal.

- **Fuente:** GroupLens, University of Minnesota — `https://files.grouplens.org/datasets/movielens/ml-100k.zip`
- **Cita del dataset:** Harper, F. M. & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context*. ACM TiiS 5(4):19.
- **Licencia:** uso libre para investigación y docencia (exige citar a Harper & Konstan 2015; no redistribución comercial). El.zip fuente es de descarga abierta.
- **Peso:** ratings CSV 2,08 MB, movies CSV 0,14 MB (<25 MB) → base **completa** en el repo (no se necesita muestra).

### 1a. `ml100k_ratings.csv` — 100.000 filas × 4 columnas (943 usuarios × 1682 películas)
SHA256: `ff8ecf3b80823ffa57f060334a671fde0df2a4c9d3760d1101e5e3ee19c1810c`

| Columna | Tipo | Significado (negocio) |
|---|---|---|
| `user_id` | entero 1-943 | Identificador del usuario que califica. Eje «cliente» de la matriz usuario-ítem. |
| `item_id` | entero 1-1682 | Identificador de la película calificada. Eje «producto»; se cruza con `movie_id` de movies. |
| `rating` | entero 1-5 | Calificación explícita (1 = no gustó, 5 = gustó mucho). Es lo que estima el recomendador. |
| `timestamp` | entero (Unix, seg.) | Momento de la calificación. Habilita la **partición temporal** (evaluar «hacia el futuro»). |

- **Dispersión (sparsity):** 93,7 % (solo el 6,3 % de las 943×1682 celdas tiene rating) — el problema central de la sesión.
- **Sin faltantes**; cada usuario calificó ≥20 películas (garantía de diseño de MovieLens).

### 1b. `ml100k_movies.csv` — 1682 filas × 22 columnas
SHA256: `f6f6427695d2f425e9ffa165f9b1057f2f5f5da9ae04e763a1af1afcab10ee14`

| Columna | Tipo | Significado (negocio) |
|---|---|---|
| `movie_id` | entero 1-1682 | Clave de la película; se une a `item_id` de los ratings. |
| `title` | texto | Título con año (p. ej. «Toy Story (1995)»). |
| `release_date` | texto | Fecha de estreno. |
| `unknown` … `Western` | 19 flags 0/1 | Indicadores de **género** (Action, Comedy, Drama, …). Son los **atributos de contenido** para el filtrado basado en contenido (perfil de gustos por género). Una película puede tener varios géneros. |

---

## 2. Online Retail II — recomendador de negocio (feedback implícito)

**Rol:** caso de e-commerce (cross/up-selling). **Feedback implícito:** no hay rating; la señal es la **compra** (cantidad comprada de cada producto por cliente). No se calcula RMSE (no hay nota que predecir): se evalúa con top-k.

- **Fuente:** UCI Machine Learning Repository, id **502** — `https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip`
- **Cita:** Chen, D. (2019). *Online Retail II*. UCI Machine Learning Repository. Licencia **CC BY 4.0**.
- **Regla OneDrive:** el.xlsx COMPLETO (~45,6 MB) **NUNCA se guarda en OneDrive**; solo la **muestra** (≤1 MB).

### `online_retail_reco_muestra.csv` — 8829 filas × 8 columnas (muestra de 80 clientes, rs=42)
SHA256: `f0ef2267cba1cf7e665aee827100b821cdb8a37478a4de501b2fcc2d3f3bc632` — 0,795 MB

| Columna | Tipo | Significado (negocio) |
|---|---|---|
| `Invoice` | texto | Nº de factura/ticket (una compra puede tener varias líneas). |
| `InvoiceDate` | fecha-hora | Momento de la compra. |
| `StockCode` | texto | Código del producto. Eje «ítem» de la matriz cliente-producto. |
| `Description` | texto | Nombre del producto (para leer las recomendaciones en lenguaje de negocio). |
| `Quantity` | entero (>0) | Unidades compradas en la línea. Es la **señal de feedback implícito** (confianza); se agrega por (cliente, producto). |
| `Price` | decimal (>0) | Precio unitario (GBP). Convierte el top-N en **ingreso potencial**. |
| `Customer ID` | entero | Identificador del cliente. Eje «usuario» de la matriz. |
| `Country` | texto | País del cliente. |

---

## Procedencia
- Ambos datasets ya están verificados (06/08/2026). `descargar_datos.py` comprueba si están disponibles localmente y, si no, los descarga de GroupLens (MovieLens) o del repo del curso (muestra de retail).
- La muestra de Online Retail II conserva `StockCode`+`Description` para poder construir la matriz cliente-producto y leer las recomendaciones.
