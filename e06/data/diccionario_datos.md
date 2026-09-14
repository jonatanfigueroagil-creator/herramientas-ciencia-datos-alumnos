# Diccionario de datos — Sesión EPE E6 (Pronosticar el futuro y entender el porqué)

> Fecha de verificación de fuentes: **06/08/2026**. Datos obtenidos y validados
> programáticamente con `descargar_datos.py` (esquema + nº de filas). Reglas: español
> impersonal, fechas DD/MM/YYYY, UTF-8.
>
> Los tres casos usan datasets ya verificados para este curso: pronóstico,
> correlación/causalidad y supervivencia. Los CSV se copian de una fuente ya
> verificada (o se obtienen del mismo mirror abierto) para mantener esta
> carpeta portable (todos < 1 MB).

---

## 1. `store1_item1.csv` — Dataset de PRONÓSTICO (Store Item Demand) — REÚSO de S11

- **Rol:** pronóstico de demanda mensual de un producto.
- **Contexto de negocio:** el reto *Store Item Demand Forecasting* contiene ventas diarias de
 50 productos en 10 tiendas (2013–2017). Para EPE se usa una **serie única y clara** —**tienda 1,
 producto 1**— que en el cuaderno se **agrega a frecuencia mensual** (60 meses) para leer tendencia
 y estacionalidad.
- **Procedencia canónica:** Kaggle — `c/demand-forecasting-kernels-only` (`train.csv`, 913 000 filas).
- **Reúso / mirror abierto** (Kaggle exige login; decisión de curso «mirrors abiertos por sesión»):
 la base completa vive en `Sesiones/S11_series_temporales/data/train.csv`; `descargar_datos.py`
 **filtra tienda 1 / producto 1** de esa base (o, si no está, la descarga del mismo mirror abierto byte-idéntico que usó S11: `raw.githubusercontent.com/jgonzalezab/Store-Item-Demand-Forecasting`).
- **Licencia:** dataset de competencia Kaggle; uso educativo.
- **Tamaño:** 1 826 filas × 4 columnas — ≈34 KB (muestra portable; la base completa de 913 000 filas no se duplica en OneDrive) — **SHA256** `eac47a67b6c4e320…` (registrado por el script).

| Columna | Tipo | Significado de negocio |
|---|---|---|
| `date` | fecha (diaria) | Día de la venta (2013-01-01 … 2017-12-31). |
| `store` | entero | Tienda (aquí siempre 1). |
| `item` | entero | Producto (aquí siempre 1). |
| `sales` | entero | **Unidades vendidas** ese día (variable a pronosticar tras agregar a mes). |

> Uso en la sesión: descomposición STL (tendencia/estacionalidad/ruido), pronóstico con
> suavizamiento exponencial (Holt-Winters) y Prophet, validación honesta con holdout de 12 meses
> contra la línea base *naive estacional*, y pronóstico a 12 meses con intervalo del 95 %.

---

## 2. `retencion_observacional.csv` — Dataset de CAUSALIDAD (simulado) — REÚSO de S12

- **Rol:** ilustrar correlación ≠ causalidad y el efecto de un **confusor**.
- **Contexto:** una empresa de suscripción lanza una **campaña de retención**. Los datos son
 **observacionales** (no aleatorizados): marketing eligió a quién contactar según la actividad
 previa (`engagement`), que **también** predice el valor futuro → es un **confusor** que sesga la
 comparación ingenua **al alza**.
- **Procedencia:** dataset **simulado** con un DGP (proceso generador de datos) conocido y
 documentado. `descargar_datos.py` busca primero una copia ya verificada o lo regenera
 inline con el **mismo DGP y semilla** (`seed=20260719`,
 `value = 50 + 8·campaign + 20·engagement + 3·tenure + N(0, 8)`).
- **Ventaja didáctica (clave en EPE):** al ser un escenario de diseño, el **efecto verdadero es
 conocido** (**+8,00 USD**), lo que permite mostrar *cuánto se equivoca* la comparación ingenua.
- **Tamaño:** 5 000 filas × 4 columnas — ≈0,29 MB.

| Columna | Tipo | Significado de negocio |
|---|---|---|
| `engagement` | número (z-score) | Actividad previa del cliente. **Confusor**: eleva la probabilidad de recibir la campaña **y** el valor. |
| `tenure` | número (meses) | Antigüedad. Afecta al valor, no a la asignación. |
| `campaign` | 0/1 | 1 = recibió la campaña de retención (**tratamiento**). |
| `value` | número (USD) | Valor incremental del cliente el próximo trimestre (**resultado**). |

> Uso en la sesión: comparación ingenua (`+23,64` USD) vs. comparar clientes similares con control
> del confusor (`+7,85` USD) vs. el efecto verdadero (`+8,00` USD) → la ingenua exagera ~3×. El A/B
> (asignación al azar) haría innecesario el ajuste: es el estándar de oro.

---

## 3. `telco_churn.csv` — Dataset de SUPERVIVENCIA (Telco Customer Churn) — REÚSO de S13

- **Rol:** curva de retención (tiempo-hasta-evento) y CLV por segmento.
- **Contexto:** 7 043 clientes de una telco; `tenure` (meses de antigüedad) es el **tiempo hasta el
 evento churn**, `Churn` indica si el cliente se dio de baja. Los clientes **activos** (sin churn a la fecha de corte) son datos **censurados por la derecha**.
- **Procedencia canónica:** Kaggle — `blastchar/telco-customer-churn` (IBM sample).
- **Reúso / mirror abierto:** se **copia de** `Sesiones/S13_supervivencia/data/telco_churn.csv`; si no
 está, se descarga del mismo mirror byte-idéntico que usó S13
 (`raw.githubusercontent.com/treselle-systems/customer_churn_analysis`).
- **Licencia:** IBM sample dataset de uso educativo.
- **Tamaño:** 7 043 filas × 21 columnas — ≈0,97 MB. En el cuaderno se descartan 11 clientes con
 `tenure = 0` (churn 26,6 % → 1 869 eventos y 5 163 censurados sobre 7 032).

| Columna (usada) | Tipo | Significado de negocio |
|---|---|---|
| `customerID` | texto | Identificador del cliente. |
| `tenure` | entero (meses) | **Tiempo** hasta el evento (antigüedad). |
| `Contract` | texto | **Segmento**: `Month-to-month` / `One year` / `Two year`. |
| `MonthlyCharges` | número (USD) | Cargo mensual (margen por periodo para el CLV). |
| `TotalCharges` | número (USD) | Cargo total acumulado (11 celdas en blanco → clientes con `tenure=0`). |
| `Churn` | texto (Yes/No) | **Evento**: `Yes` = el cliente se dio de baja. |

> Uso en la sesión: curvas de Kaplan-Meier por tipo de contrato, retención `S(t)` a 12/24/72 meses,
> prueba de log-rank y **CLV** por segmento (`Σ m·S(t)/(1+d)ᵗ`, d = 1 %/mes). Hallazgo: mes-a-mes
> tiene el **mayor margen mensual** pero el **menor CLV** (la retención domina el valor).

---

## 4. Notas de verificación y fallbacks

- **`store1_item1`:** se filtra de la base completa de S11 (o de su mirror abierto). El esquema
 (4 columnas) y el nº de filas (1 826 días) se validan en `descargar_datos.py`.
- **`retencion_observacional`:** se copia de S12; si falta, se regenera con el DGP documentado (mismo resultado, semilla fija).
- **`telco_churn`:** se copia de S13; si falta, mirror abierto byte-idéntico.
- **Idempotencia:** `descargar_datos.py` no re-obtiene un archivo si ya existe y valida su esquema.
- **Reproducibilidad Colab:** el propio cuaderno incluye un respaldo que obtiene la muestra del repo,
 Telco del mirror abierto y regenera el escenario de retención si la carpeta `data/` no está.
