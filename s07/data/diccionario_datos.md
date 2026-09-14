# Diccionario de datos — S07 (Clustering + detección de anomalías)

> Verificación real (descarga/carga programática + validación de esquema) el **18/07/2026** con el venv del curso (python), Python 3.13.5, scikit-learn 1.6.1. Genera y valida todo `descargar_datos.py` (idempotente, local + Colab, checksums SHA256). El conjunto completo de Online Retail II (~45.6 MB) **nunca se guarda en OneDrive**: solo su muestra ≤1 MB.

## Resumen de los conjuntos

| Archivo (en `data/`) | Rol | Filas × cols | Vía | Peso | SHA256 |
|---|---|---|---|---|---|
| `mammography.csv` | Replicación anomalías (Liu 2008) | 11.183 × 7 | OpenML `data_id=310` | 0,79 MB | `5b5e94f3cd7d6650…884619a4` |
| `shuttle_odds.csv` | Replicación anomalías (Liu 2008), 2.º dataset | 49.097 × 10 | OpenML `data_id=40685` + preproc ODDS | 1,28 MB | `8f16502e5d6e1217…8fa3f21e` |
| `online_retail_II_muestra.csv` | Negocio / laboratorio RFM | 18.125 × 6 (140 clientes) | UCI id 502 (muestra ≤1 MB) | 0,93 MB | `96b678af3bebf0d6…c58e0aef` |

El conjunto **completo** de Online Retail II (~1,07 M filas; xlsx 45,6 MB, SHA256 del zip `572e36277c2390fb…08e67bfb`) se descarga en runtime a un directorio temporal del sistema (fuera de OneDrive) y no se conserva.

---

## 1) MAMMOGRAPHY — `mammography.csv` (replicación Isolation Forest)

- **Procedencia:** benchmark **ODDS** (Outlier Detection DataSets, Stony Brook). El sitio original `http://odds.cs.stonybrook.edu/` tiene el **certificado SSL roto** (verificado 18/07/2026) → se resuelve por **OpenML `data_id=310`** (`fetch_openml`, nombre `mammography`). Origen primario del dato: Woods et al. (1993), mamografías digitalizadas; las **calcificaciones** son la clase minoritaria (anomalía).
- **Paper de la réplica:** Liu, F.T., Ting, K.M. & Zhou, Z.-H. (2008). *Isolation Forest*. Proc. IEEE ICDM 2008.
- **Licencia:** uso académico/investigación (ODDS/UCI). Sin restricción para docencia.
- **Esquema:** 6 features numéricas anonimizadas + 1 etiqueta binaria.

| Variable | Tipo | Significado |
|---|---|---|
| `attr1`…`attr6` | numérica (float) | 6 atributos normalizados/anonimizados que describen regiones de la mamografía (forma, textura, área de la calcificación). En el benchmark ODDS vienen sin nombre semántico. |
| `anomalia` | binaria {0,1} | **1 = calcificación (anomalía)**, 0 = tejido normal. 260 anomalías sobre 11.183 = **2,32 %**. |

## 2) SHUTTLE-ODDS — `shuttle_odds.csv` (replicación Isolation Forest, 2.º dataset)

- **Procedencia:** Statlog (Shuttle) de UCI, mirroreado en **OpenML `data_id=40685`** (58.000 × 9, 7 clases). Se aplica el **preprocesamiento canónico de ODDS** para deteccion de anomalías:
 - se **descarta la clase 4**;
 - la **clase 1** es el inlier (normal);
 - las clases pequeñas **{2, 3, 5, 6, 7}** se combinan como **anomalías**.
 - Resultado: **49.097 filas**, 3.511 anomalías = **7,15 %**.
- **Paper de la réplica:** Liu, Ting & Zhou (2008) — reporta AUC de Isolation Forest ≈ 0,99 en Shuttle. Contraste con LOF (Breunig et al. 2000).
- **Licencia:** Statlog/UCI, uso académico.
- **Esquema:** 9 features numéricas (telemetría del transbordador, sin nombre semántico en la fuente) + etiqueta binaria.

| Variable | Tipo | Significado |
|---|---|---|
| `A1`…`A9` | numérica (int) | 9 lecturas numéricas de telemetría del transbordador espacial (la primera suele ser tiempo). Sin nombres publicados en la fuente. |
| `anomalia` | binaria {0,1} | **1 = clase rara {2,3,5,6,7} (anomalía)**, 0 = clase 1 (operación normal). 3.511 / 49.097 = **7,15 %**. |

## 3) ONLINE RETAIL II — `online_retail_II_muestra.csv` (negocio, RFM)

- **Procedencia:** UCI **id 502**, https://archive.ics.uci.edu/dataset/502/online+retail+ii (zip directo `…/static/public/502/online+retail+ii.zip`). Transacciones reales de un e-commerce británico de regalos, 01/12/2009 – 09/12/2011.
- **Cita:** Chen, D. (2019). *Online Retail II*. UCI ML Repository. https://doi.org/10.24432/C5CG6D
- **Licencia:** **Creative Commons Attribution 4.0 (CC BY 4.0)**.
- **Completo:** dos hojas (`Year 2009-2010` 525.462 filas, `Year 2010-2011` 541.911 filas) → 1.067.371 filas × 8 columnas, xlsx 45,6 MB. **No se guarda en OneDrive.**
- **Limpieza aplicada** (en `cargar_retail_completo_limpio`), documentada para el laboratorio:
 1. quitar filas sin `Customer ID` (824.364 quedan);
 2. quitar **cancelaciones** (facturas cuyo `Invoice` empieza por `C`);
 3. quitar **devoluciones / cantidades no positivas** (`Quantity` ≤ 0);
 4. quitar precios no válidos (`Price` ≤ 0).
 → **805.549 filas limpias** (5.878 clientes, 41 países).
- **Muestra ≤1 MB:** 140 clientes elegidos al azar (`RandomState=42`) con **toda su historia** de compra, de modo que Recency/Frequency/Monetary quedan **exactas por cliente**. 18.125 filas, 0,93 MB. Se conservan solo las columnas necesarias para RFM (se omiten `Description` y `StockCode`, que son texto largo y aumentan de forma pronunciada el tamaño).

| Columna | Tipo | Significado / uso en RFM |
|---|---|---|
| `Invoice` | texto | N.º de factura. **Frequency** = nº de facturas distintas por cliente. |
| `InvoiceDate` | fecha-hora | Fecha de la transacción. **Recency** = días entre la última compra del cliente y la fecha de referencia (máxima fecha + 1 día). |
| `Quantity` | entero (>0) | Unidades compradas. Junto con `Price` da el importe. |
| `Price` | float (>0) | Precio unitario (GBP). |
| `Customer ID` | entero | Identificador del cliente. Clave de agregación de RFM. |
| `Country` | texto | País del cliente (contexto de segmento). |
| *(derivada)* `Amount` | float | `Quantity × Price`. **Monetary** = suma de `Amount` por cliente. |

---

## Viabilidad de la réplica — valores OBSERVADOS (no son targets)

> Los **targets con tolerancia los define la ficha de la sesión de réplica del paper** (dueño único). Aquí solo se registran los valores observados en el smoke del 18/07/2026 (venv, scikit-learn 1.6.1, `random_state=42`), como evidencia de que los datos permiten reproducir el resultado del paper.

| Dataset | AUC-ROC Isolation Forest | AUC-ROC LOF | Coherencia con la ficha de la sesión |
|---|---|---|---|
| Mammography | **0,8689** | 0,7204 | Coincide con Liu 2008 (~0,86). Sin contradicción. |
| Shuttle-ODDS | **0,9975** | 0,5235 | Coincide con Liu 2008 (~0,99). LOF muy bajo = contraste didáctico esperado (LOF, local, falla ante la estructura global de Shuttle). |

Clustering (smoke): muestra estandarizada de 1.500 filas → **Ward k=3 silhouette 0,5317**; **DBSCAN eps=0,6 / min_samples=10 → 3 clusters, 254 puntos de ruido, silhouette 0,6473** (con eps=1,5 colapsa a 1 cluster → confirma la necesidad del gráfico de k-distancia de la plantilla `guia_eps_dbscan.md`). RFM sobre la muestra (140 clientes): Ward k=3/4/5 → silhouette 0,33 / 0,34 / 0,35 (rango típico de RFM; los segmentos son separables).

**Notas de diseño para el notebook:**
- La etiqueta de anomalía existe **solo para evaluar** (AUC): Isolation Forest y LOF se ajustan **sin usarla** (no supervisado) y luego se compara el ranking de anomalía contra `anomalia`.
- Para el AUC, la elección de `contamination` no altera el ranking (solo el umbral binario), por lo que se usa `contamination='auto'`; se estandariza con `StandardScaler` antes de LOF (sensible a escala).
- Shuttle-ODDS **no** viene binario en OpenML: el notebook debe replicar el preprocesamiento ODDS (descartar clase 4; {2,3,5,6,7} = anomalía) tal como lo hace `descargar_datos.py`.
