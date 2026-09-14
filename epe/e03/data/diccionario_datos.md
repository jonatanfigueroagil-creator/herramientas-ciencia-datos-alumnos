# Diccionario de datos — Sesión EPE E3 (Demasiadas variables: regularización y PCA)

> Fecha de verificación de fuentes: **06/08/2026**. Datos obtenidos y validados
> programáticamente con `descargar_datos.py` (esquema + nº de filas + columnas clave).
> Reglas: español impersonal, fechas DD/MM/YYYY, UTF-8. Ningún dato es sintético.

---

## 1. `communities.csv` — Dataset de REGULARIZACIÓN (alta dimensión)

- **Rol:** caso de «demasiadas variables» (muchos predictores correlacionados) para comparar OLS vs. RIDGE vs. LASSO y ver la selección automática del LASSO.
- **Contexto de negocio:** se predice la **tasa de crimen violento** (`ViolentCrimesPerPop`, por 100 000 habitantes) de **1 994 comunidades** de EE. UU. a partir de ~100 indicadores socioeconómicos (censo 1990) y policiales (LEMAS 1990), todos **normalizados a [0, 1]**.
- **Procedencia canónica:** UCI Machine Learning Repository, id **183** (Redmond, M., 2009). DOI 10.24432/C53W3X. Combina censo 1990 de EE. UU., LEMAS 1990 y UCR 1995 del FBI.
- **Mirror abierto usado:** `https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data`. En local, `descargar_datos.py` **copia una versión local ya verificada** para evitar re-descargarlo.
- **Licencia:** uso educativo (repositorio UCI, acceso abierto).
- **Tamaño:** 1 994 filas × 128 columnas, ≈1.05 MB (< 25 MB → se conserva en la carpeta), **SHA256** `f6d3ff2ca832aafc0d1b7f935d6f39647bd3f3f17436c60ff3009ccf99352460`.
- **Estructura:** **5 columnas no predictivas** (`state`, `county`, `community`, `communityname`, `fold`) + **122 predictores** numéricos en [0, 1] + la variable objetivo `ViolentCrimesPerPop`.
- **Calidad (verificada 06/08/2026):** hay **faltantes** codificados originalmente como `?` → convertidos a `NaN`. Un bloque de columnas policiales (p. ej. `Lemas*`, `Polic*`) tiene ~84 % de faltantes; en la sesión se **imputan por la media** como paso de preparación (rebaja EPE: imputación simple, sin discutir mecanismos avanzados).

| Grupo de columnas | Tipo | Significado de negocio |
|---|---|---|
| `state`, `county`, `community`, `communityname`, `fold` | id / texto | Identificadores geográficos y de partición. **No** son variables de análisis. |
| ~122 predictores socioeconómicos y policiales | cuantitativas [0,1] | Ingreso, empleo, educación, estructura familiar, vivienda, inmigración, urbanización, indicadores policiales. Todos normalizados. |
| `ViolentCrimesPerPop` | cuantitativa [0,1] | **Variable objetivo:** crímenes violentos por 100 000 hab. (normalizada). |

> Uso en la sesión: separar predictores del objetivo, imputar faltantes por la media, estandarizar, partir en train/test (30 %) y comparar **OLS / RIDGE / LASSO** (λ elegido por validación cruzada), al leer el error en datos de prueba y las variables que **conserva** el LASSO.

---

## 2. `wine.csv` — Dataset de PCA (comprimir / visualizar)

- **Rol:** caso para reducir dimensiones con PCA (varianza explicada) y **visualizar** en 2D.
- **Contexto de negocio:** **178 vinos** de **tres cultivares** de una misma región de Italia, descritos por **13 medidas químicas** (alcohol, fenoles, flavonoides, intensidad de color, tono, prolina, etc.). El PCA resume las 13 variables en pocas componentes y separa los cultivares en un plano.
- **Procedencia canónica:** UCI Machine Learning Repository, id **109** (Forina et al.); integrado en scikit-learn como `sklearn.datasets.load_wine`.
- **Mirror / generación:** en local `descargar_datos.py` **copia una versión local ya verificada**; como respaldo (Colab) se regenera con `load_wine` (mismo conjunto, mismo esquema).
- **Licencia:** uso educativo (dataset clásico UCI, acceso abierto).
- **Tamaño:** 178 filas × 14 columnas, ≈12 KB (< 25 MB → se conserva en la carpeta), **SHA256** `2cce9d95d7d95165a591b84741d15d3300f6862a135a28e3601249d42f303056`.
- **Calidad:** **0 valores faltantes**; escalas muy dispares entre variables (la `proline` va en cientos; el `hue`, en decimales) → **obliga a estandarizar** antes del PCA (lección central del bloque).

| Columna | Tipo | Significado de negocio |
|---|---|---|
| `alcohol`, `malic_acid`, `ash`, `alcalinity_of_ash`, `magnesium`, `total_phenols`, `flavanoids`, `nonflavanoid_phenols`, `proanthocyanins`, `color_intensity`, `hue`, `od280/od315_of_diluted_wines`, `proline` | cuantitativas | Las **13 medidas químicas** del vino (escalas heterogéneas). |
| `target` | entero (0/1/2) | **Cultivar** de origen. Se usa solo para **colorear** el gráfico 2D; el PCA es **no supervisado** (no lo usa para reducir). |

> Uso en la sesión: comparar la convención **covarianza** (datos crudos) vs. **correlación** (estandarizado) —crudo, PC1 «explica» 99.8 % por artefacto de escala; estandarizado, PC1 ≈ 36 % y hacen falta 8 componentes para el 90 %—, elegir el nº de componentes por varianza acumulada, nombrar PC1/PC2 con sus cargas y visualizar los 178 vinos en 2D.

---

## 3. Notas de verificación y reúso

- **Idempotencia:** `descargar_datos.py` no re-obtiene un archivo si ya existe con el esquema correcto (filas × columnas + columnas clave).
- **Copia local:** en local se **copian** versiones ya verificadas de los CSV, para evitar re-descargarlos (mismos checksums de origen). Si no estuvieran disponibles, se obtienen del mirror abierto de UCI (communities) o de `load_wine` (wine).
- **Reproducibilidad Colab:** el propio cuaderno incluye un respaldo que descarga Communities and Crime del mirror de UCI y genera Wine con `load_wine` si la carpeta `data/` no está presente.
- **Sin datasets >25 MB:** ambos archivos son pequeños; no aplica la regla de exclusión de OneDrive ni se necesita muestra.
