# Diccionario de datos — Sesión EPE E4 (Segmentar clientes y encontrar patrones)

> Fecha de verificación de fuentes: **06/08/2026**. Datos obtenidos y validados
> programáticamente con `descargar_datos.py` (esquema + nº de filas + checksum SHA256).
> Reglas: español impersonal, fechas DD/MM/YYYY, UTF-8. Ningún dato es sintético.
>
> Toda la sesión trabaja con **Online Retail II** (UCI id 502). Aquí NO se descarga
> el completo (~45.6 MB): se usan **dos muestras ≤1 MB**, cada una preparada
> según su unidad de análisis (cliente vs. factura).

---

## Fuente común: Online Retail II (UCI id 502)

- **Cita:** Chen, D. (2019). *Online Retail II*. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
- **Contexto de negocio:** transacciones reales de un e-commerce británico de artículos de regalo, 01/12/2009 – 09/12/2011.
- **Licencia:** **Creative Commons Attribution 4.0 (CC BY 4.0)**.
- **Completo:** 1 067 371 filas × 8 columnas, xlsx 45,6 MB (dos hojas). **Nunca se guarda en OneDrive**; se descarga en runtime a un temporal del sistema y se conserva solo la muestra ≤1 MB.

---

## 1. `online_retail_II_rfm_muestra.csv` — segmentación por CLIENTE (RFM)

- **Rol:** base de la **segmentación de clientes** (análisis RFM + k-means).
- **Procedencia:** muestra ya verificada, copiada aquí con nombre explícito.
- **Cómo se muestreó:** 140 clientes elegidos al azar (`RandomState=42`) con **toda su historia** de compra, de modo que Recencia, Frecuencia y Monto quedan **exactas por cliente**. Se omiten `Description` y `StockCode` (texto largo que multiplica el tamaño del archivo); no se necesitan para RFM.
- **Tamaño:** 18 125 filas × 6 columnas · ≈0,93 MB (< 25 MB → se conserva) · **SHA256** `96b678af3bebf0d6277e0e101c7fca0745b37c6d080a2158df0adc3bc58e0aef`.
- **Limpieza aplicada:** sin `Customer ID` nulos, sin cancelaciones (`Invoice` que empieza por `C`), sin `Quantity ≤ 0` ni `Price ≤ 0`.

| Columna | Tipo | Significado / uso en RFM |
|---|---|---|
| `Invoice` | texto | N.º de factura. **Frequency** = nº de facturas distintas por cliente. |
| `InvoiceDate` | fecha-hora | Fecha de la transacción. **Recency** = días entre la última compra del cliente y la fecha de referencia (máxima fecha + 1 día). |
| `Quantity` | entero (>0) | Unidades compradas. Con `Price` da el importe. |
| `Price` | float (>0) | Precio unitario (GBP). |
| `Customer ID` | entero | Identificador del cliente. **Clave de agregación** de RFM. |
| `Country` | texto | País del cliente (contexto de segmento). |
| *(derivada)* `Amount` | float | `Quantity × Price`. **Monetary** = suma de `Amount` por cliente. |

> Uso en la sesión: calcular R, F, M por cliente; estandarizar (con `log` en F y M por su cola larga); agrupar con **k-means**; elegir el nº de segmentos con el **coeficiente de silueta**; **perfilar** y nombrar cada segmento; y marcar clientes **inusuales** (detección de anomalías, a nivel de intuición).

---

## 2. `online_retail_baskets_muestra.csv` — market basket por FACTURA

- **Rol:** base de las **reglas de asociación** (canasta de mercado / cross-selling).
- **Procedencia:** muestra ya verificada.
- **Cómo se muestreó:** 800 facturas elegidas al azar (`RandomState=42`) con **todas sus líneas** → cada factura es una **canasta completa** (no truncada). Conserva `Description` (el ítem legible que se necesita para leer reglas).
- **Tamaño:** 16 979 filas × 4 columnas · ≈0,95 MB (< 25 MB → se conserva) · **SHA256** `a3745101de08f44a8fc67167e9075cb12427a776b53e7d0cdc503a78ddae2760`.

| Columna | Tipo | Significado de negocio |
|---|---|---|
| `Invoice` | texto (6 díg.) | N.º de **factura/ticket** = identificador de la **canasta**. Todas las filas con la misma `Invoice` son un mismo ticket. |
| `StockCode` | texto | Código interno del **producto** (SKU). |
| `Description` | texto | **Nombre del producto** (el ítem de la canasta; base de las reglas legibles, p. ej. «GREEN REGENCY TEACUP AND SAUCER»). |
| `Country` | texto | País del cliente (mayoría «United Kingdom»). |

> **Diferencia clave con la muestra 1.** La muestra RFM se armó **por cliente** (para segmentar) y **no** trae `Description`. Esta muestra se armó **por factura** y **sí** conserva `Description` (el nombre del producto). No son intercambiables: una segmenta clientes, la otra encuentra productos que se compran juntos.

> Uso en la sesión: agrupar las líneas por `Invoice` → lista de productos por ticket; codificar en **one-hot booleano** por factura (`TransactionEncoder`); minar **itemsets frecuentes** con **Apriori** (`min_support`); generar **reglas** con `association_rules` y leer **soporte, confianza y lift**; filtrar triviales (lift ≈ 1) y proponer **cross-selling**.

---

## 3. Notas de verificación

- **Idempotencia:** `descargar_datos.py` no re-obtiene un archivo si ya existe y su SHA256 coincide.
- **Copia local:** en local copia una versión ya verificada de la muestra; si no está, la descarga del **mirror abierto del repositorio del curso** (GitHub raw). En Colab descarga del mirror.
- **Sin API keys:** ni Kaggle ni credenciales; todo es acceso abierto (CC BY 4.0).
- **Reproducibilidad Colab:** el propio cuaderno incluye un respaldo que descarga los CSV del mismo mirror si la carpeta `data/` no está presente.
- **Opciones no usadas (mención EPE):** la sesión no incluye el completo de Online Retail II ni datasets de anomalías adicionales; la detección de anomalías se ilustra a nivel de intuición sobre los mismos clientes RFM.
