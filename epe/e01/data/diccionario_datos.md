# Diccionario de datos — Sesión EPE E1 (Fundamentos: CRISP-DM + EDA + A/B testing)

> Fecha de verificación de fuentes: **06/08/2026**. Datos obtenidos y validados
> programáticamente con `descargar_datos.py` (esquema + nº de filas + checksum SHA256).
> Reglas: español impersonal, fechas DD/MM/YYYY, UTF-8. Ningún dato es sintético.
>
> Ambos archivos del caso A/B *Cookie Cats* (la muestra y la base completa) ya
> fueron verificados para este curso —mismo esquema, mismo número de filas y
> mismo checksum SHA256.

---

## 1. `Mall_Customers.csv` — Dataset de EXPLORACIÓN (EDA)

- **Rol:** caso de exploración de datos (perfiles de clientes de un centro comercial).
- **Contexto de negocio:** un centro comercial (mall) caracteriza a sus clientes con
 edad, ingreso anual y un **puntaje de gasto** (1–100) que asigna según el
 comportamiento de compra, para orientar campañas y segmentación.
- **Procedencia canónica:** Kaggle — `vjchoudhary7/customer-segmentation-tutorial-in-python`.
- **Mirror abierto usado** (sin token de Kaggle; decisión de curso «mirrors abiertos por sesión»):
 `https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv`
 (esquema y nº de filas idénticos al original de Kaggle; la columna de sexo se normaliza a `Gender`). Fallback verificado: mirror `SteffiPeTaffy/machineLearningAZ`
 (columna `Genre`, renombrada a `Gender` por el script).
- **Licencia:** el dataset original de Kaggle no declara licencia explícita; uso educativo.
- **Tamaño:** 200 filas × 5 columnas, ≈4 KB (< 25 MB → se conserva en la carpeta),
 **SHA256** `416a4f62a8f33841a16e1db5297f0e05d3b65b05e73632cedae58e739e1e1f77`.
- **Calidad (verificada 06/08/2026):** **0 valores faltantes**; **2 atípicos** en el
 ingreso anual (los dos clientes de mayor renta, ≈137 k$), genuinos, se conservan.
- **Balance:** `Female` = 112 — `Male` = 88.

| Columna | Tipo | Significado de negocio |
|---|---|---|
| `CustomerID` | entero | Identificador único del cliente. No es una variable de análisis. |
| `Gender` | texto (nominal) | Sexo del cliente (`Male` / `Female`). |
| `Age` | entero (cuantitativa) | Edad del cliente en años. |
| `Annual Income (k$)` | entero (cuantitativa) | Ingreso anual del cliente en **miles de dólares**. |
| `Spending Score (1-100)` | entero (cuantitativa) | **Puntaje de gasto** (1 a 100) asignado por el mall según el comportamiento de compra del cliente. Métrica de valor. |

> Uso en la sesión: EDA (tipos, descriptiva, faltantes/atípicos, histograma, boxplot,> barras, dispersión, mapa de correlación) y comparación de dos grupos (prueba t del> gasto por edad; chi-cuadrado de sexo × nivel de gasto).

---

## 2. `cookie_cats.csv` — Dataset de NEGOCIO (experimento A/B) — REÚSO de S02

- **Rol:** caso de negocio del experimento A/B (retención de un juego móvil).
- **Contexto:** *Cookie Cats* (Tactile Entertainment), juego móvil de puzles. El
 experimento mueve la primera «puerta» (gate) del nivel 30 al nivel 40 y mide el
 impacto en la retención de jugadores.
- **Procedencia canónica:** Kaggle — `yufengsui/mobile-games-ab-testing`.
- **Mirror abierto usado** (el mismo que S02):
 `https://raw.githubusercontent.com/ryanschaub/Mobile-Games-A-B-Testing-with-Cookie-Cats/master/cookie_cats.csv`
 (fork del proyecto DataCamp «Mobile Games A/B Testing with Cookie Cats»; esquema y nº de filas idénticos al original de Kaggle). En local, `descargar_datos.py` **copia
 la base ya verificada de S02** para evitar re-descargarla.
- **Licencia:** el dataset original de Kaggle no declara licencia explícita; uso educativo.
- **Tamaño:** 90 189 filas × 5 columnas, ≈2.58 MB (< 25 MB → se conserva en la carpeta),
 **SHA256** `5ab54d761fbddcd50de7b88e4eaf7837cba4569474f50c043a4d17ee342c46bd`.
- **Muestra:** `cookie_cats_muestra.csv` (3 000 filas, ≈91 KB) para vistazos rápidos y
 portables; **reutilizada de S02** (misma semilla, `random_state=42`).
- **Balance de grupos (base completa):** `gate_30` = 44 700 — `gate_40` = 45 489.

| Columna | Tipo | Significado de negocio |
|---|---|---|
| `userid` | entero | Identificador único del jugador. |
| `version` | texto | **Grupo del experimento A/B**: `gate_30` (control, puerta en nivel 30) o `gate_40` (tratamiento, puerta en nivel 40). |
| `sum_gamerounds` | entero | **Partidas jugadas** en los primeros 14 días tras instalar (métrica de engagement, muy sesgada; hay un valor extremo de 49 854). |
| `retention_1` | booleano | **Retención día 1**: el jugador volvió a jugar 1 día después de instalar. |
| `retention_7` | booleano | **Retención día 7**: el jugador volvió a jugar 7 días después de instalar. **Métrica primaria de decisión (OEC)** del A/B. |

> Uso en la sesión: comparar la retención (proporciones) entre `gate_30` y `gate_40`
> en día 1 y día 7, calcular el **lift**, distinguir significancia estadística de
> práctica y emitir una recomendación. La muestra de 3 000 se usa como ejemplo de que
> un experimento **subdimensionado** no permite decidir (poder estadístico).

---

## 3. Notas de verificación y fallbacks

- **`Mall_Customers`:** el mirror primario responde 200 y validó esquema/nº de filas.
 Si cayera, `descargar_datos.py` usa el mirror `SteffiPeTaffy` (renombra `Genre`→`Gender`).
 Vía manual: Kaggle `vjchoudhary7/customer-segmentation-tutorial-in-python`.
- **`cookie_cats`:** en local se copia la base ya verificada de
 `Sesiones/S02_eda_inferencia_ab/data/cookie_cats.csv` (idéntico checksum); si no está,
 se descarga del mirror de GitHub. La muestra se copia de S02.
- **Idempotencia:** `descargar_datos.py` no re-obtiene un archivo si ya existe (y, para la base completa de Cookie Cats, si su SHA256 coincide).
- **Reproducibilidad Colab:** el propio cuaderno incluye un respaldo que descarga los
 CSV de los mismos mirrors abiertos si la carpeta `data/` no está presente.
