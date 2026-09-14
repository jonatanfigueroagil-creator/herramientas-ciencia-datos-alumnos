# Diccionario de datos — S14 Sistemas de recomendacion

> Fecha de verificacion: **06/08/2026**. Verificacion REAL (descarga/carga programatica y validacion de filas x columnas), no solo HTTP 200. Generado y validado con `descargar_datos.py` sobre el venv del curso (la matriz de versiones certificada del curso). Idempotente y reproducible en local y en Colab.

Este archivo describe los **dos** datasets de la sesion: uno de **replicacion** (MovieLens 100k, base del paper seminal) y uno de **negocio** (Online Retail II, feedback implicito). Los **targets cuantitativos con tolerancia NO se definen aqui**: son propiedad declarado para la sesión de réplica del paper (dueño: `investigador-tema`).

---

## 1. Replicacion — MovieLens 100k

**Rol:** base de la replica del paper seminal de **factorizacion de matrices**: Koren, Bell & Volinsky (2009), *Matrix Factorization Techniques for Recommender Systems*, IEEE Computer 42(8).

**Base original: SUSTITUTO — justificado.** El dataset original sobre el que Koren-Bell-Volinsky desarrollaron la tecnica fue el **Netflix Prize**, que **YA NO se distribuye** (retirado tras el litigio de privacidad de 2010). **MovieLens 100k** (GroupLens) es el **sustituto canonico** con el que la literatura reproduce la factorizacion SVD sobre datos de rating explicito 1-5. Se declara y etiqueta como sustituto en todo el material.

- **Fuente:** GroupLens, University of Minnesota — `https://files.grouplens.org/datasets/movielens/ml-100k.zip`
- **Enlace de referencia:** https://grouplens.org/datasets/movielens/100k/ (verificado HTTP-200 en catalogo S00).
- **Cita del dataset:** Harper, F. M. & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context*. ACM TiiS 5(4):19.
- **Licencia / uso:** licencia de uso de GroupLens — libre para investigacion y docencia; se exige citar a Harper & Konstan (2015); no redistribuir con fines comerciales. El.zip fuente es de descarga abierta.
- **SHA256 del.zip fuente:** `50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229`
- **Peso:**.zip 4,9 MB; ratings CSV 2,03 MB (<25 MB) -> la base **completa** de replica se conserva en el repo (no se necesita muestra).
- **Fallback documentado:** `ml-latest-small` (`https://files.grouplens.org/datasets/movielens/ml-latest-small.zip`, SHA256 `696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436`, 0,98 MB) — Advertencia: **esquema distinto** (ratings 0,5-5,0 en pasos de 0,5; columnas `userId,movieId,rating,timestamp`); si se usa, recalibrar el `Reader(rating_scale=...)` y los targets.

### 1a. `ml100k_ratings.csv` — 100.000 filas x 4 columnas (943 usuarios x 1682 peliculas)
SHA256: `ff8ecf3b80823ffa57f060334a671fde0df2a4c9d3760d1101e5e3ee19c1810c`

| Columna | Tipo | Significado (negocio) |
|---|---|---|
| `user_id` | entero 1-943 | Identificador del usuario (quien califica). Es el eje "cliente" de la matriz usuario-item. |
| `item_id` | entero 1-1682 | Identificador de la pelicula calificada. Es el eje "producto"; se cruza con `movie_id` de `ml100k_movies.csv`. |
| `rating` | entero 1-5 | Calificacion explicita (feedback explicito). 1 = no gusto, 5 = gusto mucho. Es la variable a predecir en la replica SVD. |
| `timestamp` | entero (Unix, seg.) | Momento de la calificacion. Habilita la **particion temporal** (evaluar recomendaciones "hacia el futuro"). |

- **Dispersion (sparsity):** 0,937 (solo el 6,3 % de las 943x1682 celdas tiene rating) — ilustra el problema central de la sesion.
- **Sin faltantes**; cada usuario califico >=20 peliculas (garantia del diseño de MovieLens).

### 1b. `ml100k_movies.csv` — 1682 filas x 22 columnas
SHA256: `f6f6427695d2f425e9ffa165f9b1057f2f5f5da9ae04e763a1af1afcab10ee14`

| Columna | Tipo | Significado (negocio) |
|---|---|---|
| `movie_id` | entero 1-1682 | Clave de la pelicula; se une a `item_id` de los ratings. |
| `title` | texto | Titulo con año (p. ej. "Toy Story (1995)"). |
| `release_date` | texto (dd-Mmm-aaaa) | Fecha de estreno. |
| `unknown` … `Western` | 19 flags 0/1 | Indicadores de **genero** (Action, Adventure, Animation, Children, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western, y `unknown`). Son los **atributos de contenido** para el **filtrado basado en contenido** (perfil de gustos por genero). Una pelicula puede tener varios generos. |

### Viabilidad de la replica (valores OBSERVADOS — NO son targets)
Smoke reproducible (`descargar_datos.py`, particion 80/20 aleatoria, `random_state=42`):
- RMSE baseline (media global): **1,1303**
- RMSE **SVD** (100 factores, 20 epochs): **0,9352** (mejora ~17 % sobre el baseline).

> **Nota para `investigador-tema` (posible contradiccion de escala):** el RMSE historico del **Netflix Prize** (~0,8567 de la solucion ganadora) NO es comparable con MovieLens 100k (dataset y escala distintos). Los targets deben calibrarse a **MovieLens 100k** (SVD RMSE ~0,91-0,94 segun particion), no a las cifras de Netflix. El diseño es de **rating explicito 1-5** (no pareado); SVD de `surprise` reproduce la factorizacion latente de Koren-Bell-Volinsky. La particion estandar `u1.base`/`u1.test` (80/20) tambien viene en el.zip si se prefiere una particion fija a la aleatoria del smoke.

---

## 2. Negocio — Online Retail II (feedback implicito)

**Rol:** caso de negocio (up/cross-selling, retencion). **Sustituto de negocio, no replica.** Feedback **implicito**: no hay rating; la señal es la **compra** (cantidad comprada de cada producto por cliente).

- **Fuente:** UCI Machine Learning Repository, id **502** — `https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip` (misma fuente ya verificada en S07/S08).
- **Cita:** Chen, D. (2019). *Online Retail II*. UCI Machine Learning Repository. Licencia **CC BY 4.0**.
- **SHA256 del.zip fuente:** `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb`
- **Peso / regla OneDrive:** el.xlsx COMPLETO (~1,07M filas, ~45,6 MB) **NUNCA se guarda en OneDrive**; `descargar_datos.py` lo baja a un temporal del sistema solo en runtime y conserva **solo la muestra** (<=1 MB) + checksum. En Colab se descarga completo en runtime.
- **Limpieza (igual que S07):** se descartan filas sin `Customer ID`, cancelaciones (`Invoice` que empieza por "C"), `Quantity<=0` y `Price<=0`. Completo limpio: 805.549 filas, 5878 clientes, 4631 productos.

### `online_retail_reco_muestra.csv` — 8829 filas x 8 columnas (muestra de 80 clientes, rs=42)
SHA256: `f0ef2267cba1cf7e665aee827100b821cdb8a37478a4de501b2fcc2d3f3bc632` — 0,795 MB

| Columna | Tipo | Significado (negocio) |
|---|---|---|
| `Invoice` | texto | Numero de factura/ticket (una compra puede tener varias lineas). |
| `InvoiceDate` | fecha-hora | Momento de la compra. Habilita la **particion temporal** del recomendador. |
| `StockCode` | texto | Codigo del producto. Es el eje "item" de la matriz cliente-producto. |
| `Description` | texto | Nombre del producto (para leer las recomendaciones en lenguaje de negocio). |
| `Quantity` | entero (>0) | Unidades compradas en la linea. Es la **señal de feedback implicito** (confianza); se agrega por (cliente, producto). |
| `Price` | decimal (>0) | Precio unitario (GBP). Permite ponderar por valor / traducir a ingreso. |
| `Customer ID` | entero | Identificador del cliente. Es el eje "usuario" de la matriz. |
| `Country` | texto | Pais del cliente (segmentacion geografica). |

### Viabilidad del recomendador de negocio (OBSERVADO)
Matriz cliente x producto construida con `Quantity` agregada: **80 x 2235**, 5737 celdas no nulas (densidad 3,2 %). **`implicit.ALS`** (32 factores) ajusta y produce top-N por cliente sin error -> el dataset **soporta el laboratorio de feedback implicito** (Hu-Koren-Volinsky 2008). La muestra es pequeña por el limite de 1 MB del repo; el notebook en Colab puede usar el completo para escala.

---

## Procedencia y reutilizacion
- La fuente Online Retail II se **reutiliza** de S07 (`Sesiones/S07_clustering_anomalias/data/`) y S08; aqui se regenera una **muestra distinta** que conserva `StockCode`+`Description` (las muestras de S07/S08 los habian descartado: S07 era RFM por cliente, S08 cestas por factura) para poder construir la matriz cliente-producto.
- Registro en el material de referencia de la sesión (fila S14).
