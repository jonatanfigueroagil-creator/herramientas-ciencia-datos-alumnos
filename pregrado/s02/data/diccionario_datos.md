# Diccionario de datos — Sesión S02 (EDA + inferencia + A/B testing)

> Fecha de verificación de fuentes: **17/07/2026**. Datos descargados y validados
> programáticamente con `descargar_datos.py` (esquema + nº de filas + checksum SHA256).
> Reglas: español impersonal, fechas DD/MM/YYYY, UTF-8. Ningún dato es sintético.

---

## 1. `sleep.csv` — Dataset de REPLICACIÓN (paper seminal)

- **Rol:** replicación del paper seminal de la prueba t.
- **Paper:** Student (W. S. Gosset) (1908). *The Probable Error of a Mean*. **Biometrika 6(1), 1–25.** Los datos provienen del experimento farmacológico de **Cushny & Peebles (1905)** sobre el efecto de dos hipnóticos en las horas de sueño de 10 pacientes.
- **Procedencia / mirror:** Rdatasets — paquete R `datasets`, dataset `sleep`.
 `https://vincentarelbundock.github.io/Rdatasets/csv/datasets/sleep.csv`
- **Licencia:** dominio público / GPL (paquete base de R). Uso académico libre.
- **Tamaño:** 20 filas × 4 columnas, 238 bytes, **SHA256** `5563ff4da6cf477df91551d9455f905dd0090cd82948fca985eda8a5440e548f`.
- **Estructura PAREADA (clave):** formato largo. Cada uno de los **10 sujetos** (`ID` 1–10) aparece **dos veces**, una por cada fármaco (`group` 1 y 2) → 10 × 2 = 20 filas. El análisis correcto es una **prueba t pareada** sobre la diferencia intra-sujeto `extra(group 2) − extra(group 1)`.

| Columna | Tipo | Significado (negocio/experimento) |
|---|---|---|
| `rownames` | entero | Índice de fila heredado de R (1–20). No es una variable de análisis. |
| `extra` | decimal | **Horas extra de sueño** del paciente respecto a su línea base, bajo el fármaco indicado. Es la variable de respuesta. |
| `group` | entero (1/2) | **Fármaco administrado**: 1 = dextro-hyoscyamine, 2 = laevo-hyoscyamine (los dos hipnóticos comparados). |
| `ID` | entero (1–10) | **Identificador del paciente** (factor de pareo). El mismo `ID` recibe ambos fármacos. |

### Réplica — valores OBSERVADOS (smoke test 17/07/2026)
Prueba t **pareada** de `extra` (fármaco 2 − fármaco 1), n = 10 sujetos:

| Métrica | Observado | Target declarado | ¿Coincide? |
|---|---|---|---|
| Diferencia media de horas de sueño | **1.58** | 1.58 ± 0.05 | Sí |
| Estadístico t (df = 9) | **4.0621** | 4.06 ± 0.15 | Sí |
| p-valor (bilateral) | **0.002833** | 0.0028 ± 0.001 | Sí |

Herramienta que calcula: `scipy.stats.ttest_rel` (+ `numpy` y `statsmodels`). La salida
ampliada -- IC95 (0.7001, 2.4599), Cohen d combinada = 0.8322, poder = 0.6500 -- se
calcula e imprime en la celda 61 del cuaderno con scipy / numpy / statsmodels TTestPower.
`pingouin.ttest(..., paired=True)` agrupa esas mismas cifras en una tabla, pero NO esta
instalado en el venv del curso y no produce ninguna cifra del material (solo Colab).
**Advertencia:** una prueba t de **dos muestras independientes** sobre `extra` da
t ≈ 1.86 y **NO** reproduce el resultado publicado — se debe usar el pareo por `ID`.

---

## 2. `cookie_cats.csv` — Dataset de NEGOCIO (laboratorio A/B)

- **Rol:** caso de negocio del laboratorio (análisis de un A/B test real).
- **Contexto:** *Cookie Cats* (Tactile Entertainment), juego móvil de puzles. El experimento mueve la primera "puerta" (gate) del nivel 30 al nivel 40 y mide el impacto en la retención de jugadores.
- **Procedencia canónica:** Kaggle — `yufengsui/mobile-games-ab-testing`.
- **Mirror abierto usado** (sin token de Kaggle; decisión de curso "mirrors abiertos por sesión"):
 `https://raw.githubusercontent.com/ryanschaub/Mobile-Games-A-B-Testing-with-Cookie-Cats/master/cookie_cats.csv`
 (fork del proyecto DataCamp "Mobile Games A/B Testing with Cookie Cats"; esquema y nº de filas idénticos al original de Kaggle).
- **Licencia:** el dataset original de Kaggle no declara licencia explícita; uso educativo. Si se requiere procedencia oficial, descargar de Kaggle (ver `README_datos.md` si se prefiere la vía manual).
- **Tamaño:** 90 189 filas × 5 columnas, 2.58 MB (< 25 MB → se conserva en la carpeta), **SHA256** `5ab54d761fbddcd50de7b88e4eaf7837cba4569474f50c043a4d17ee342c46bd`.
- **Muestra:** `cookie_cats_muestra.csv` (3 000 filas, 91 KB) para vistazos rápidos.
- **Balance de grupos:** `gate_30` = 44 700 — `gate_40` = 45 489.

| Columna | Tipo | Significado de negocio |
|---|---|---|
| `userid` | entero | Identificador único del jugador. |
| `version` | texto | **Grupo del experimento A/B**: `gate_30` (control, puerta en nivel 30) o `gate_40` (tratamiento, puerta en nivel 40). |
| `sum_gamerounds` | entero | **Partidas jugadas** por el usuario en los primeros 14 días tras instalar. Métrica de engagement (muy sesgada a la derecha; hay un valor extremo de 49 854). |
| `retention_1` | booleano | **Retención día 1**: el jugador volvió a jugar 1 día después de instalar. |
| `retention_7` | booleano | **Retención día 7**: el jugador volvió a jugar 7 días después de instalar (métrica de decisión del A/B). |

> Uso en el laboratorio: contrastar la retención (proporciones) entre `gate_30` y
> `gate_40`, calcular tamaño muestral/poder y lift, distinguir significancia
> estadística de práctica y emitir una recomendación de negocio.

---

## 3. Notas de verificación y fallbacks

- **`sleep`:** fuente primaria (Rdatasets) respondió 200 y validó esquema/nº de filas. `descargar_datos.py` incluye además un **fallback embebido con los datos publicados de Cushny-Peebles** (no sintéticos, con cita) para garantizar reproducibilidad en Colab si el mirror no responde. Cadena de fallback prevista en la ficha de la sesión: Rdatasets ↔ statsmodels ↔ integrado.
- **`cookie_cats`:** el mirror de GitHub coincide exactamente con el original de Kaggle (90 189 filas, mismos splits de grupo). Si el mirror cayera, la vía es la descarga manual desde Kaggle (documentar en `README_datos.md`).
- **Idempotencia:** `descargar_datos.py` no re-descarga si el archivo ya existe y su SHA256 coincide.

## 4. Incidencias de entorno detectadas y RESUELTAS (17/07/2026)

- **`ydata-profiling 4.18.4` no importaba** en el venv: `ModuleNotFoundError: No module named 'pkg_resources'`. Causa: `setuptools 83.0.0` (≥81) removió `pkg_resources`. **RESUELTO:** se instaló `setuptools==80.10.2` (`pip install "setuptools<81"`); `pkg_resources` restaurado y `ProfileReport(minimal=True)` verifica que genera HTML. Pin fijado en la matriz de versiones certificada del curso (no subir setuptools a ≥81 sin migrar a `fg-data-profiling`). `pandas`, `scipy`, `numpy` y `requests` ya importaban bien. **Corrección del 15/08/2026:** `pingouin 0.6.1` **no está instalado** en el venv (ni `ydata-profiling`): ambos son opcionales y solo se instalan en Colab (celda 3 del cuaderno). Ninguna cifra de la sesión depende de ellos.
- **Desfase de versiones venv vs. documentación:** el venv real tiene `pandas 2.3.3`, **`numpy 2.5.1`**, `scipy 1.16.3` y `matplotlib 3.11.1`; `entorno.md`/el material de la sesión citaban `pandas 3.0.3` / `scipy 1.18.0` (inexacto). **RESUELTO:** la matriz de versiones certificada del curso y el cuaderno de la sesión Sección 3` reconciliados contra `pip list`. No afectó la réplica (los targets coinciden). **Corrección del 15/08/2026:** esta nota decía «numpy 2.3.5» porque se escribió el 17/07/2026, **antes** del reanclaje del 22/07/2026 que reconstruyó el venv en `Usuario`·`C:`; tras ese reanclaje el venv quedó con **numpy 2.5.1 / matplotlib 3.11.1 / Python 3.13.14**, tal como reverifica la matriz de versiones certificada del curso el 14/08/2026. Los pines de Colab del cuaderno (celda 3) siguen esas versiones, **paquete por paquete y con degradación individual**.
