# Diccionario de datos — Sesión 12 (Inferencia causal)

> Fecha de verificación: **19/07/2026**. Generado y validado con el venv del curso
> (`d:\OneDrive\Cursos\Herramientas de Ciencias de Datos\python, Python 3.13.5).
> Todos los datasets se materializan con `data/descargar_datos.py` (idempotente, con checksum SHA256).
> Los valores de DiD/ATT publicados van **ETIQUETADOS como benchmark**; los targets con tolerancia
> los fija la ficha de la sesión de réplica del paper (dueño único), no este archivo.

## Resumen de archivos

| Archivo | Rol | Filas × Cols | Tamaño | SHA256 (12) | Fuente |
|---|---|---|---|---|---|
| `card_krueger_1994.csv` | Replicación DiD | 410 × 50 | 0,07 MB | `74bf69737c51` | David Card (Berkeley) |
| `lalonde_nsw.csv` | Replicación PSM (experimental) | 445 × 11 | 0,02 MB | `878c52cc105f` | `causaldata` (Dehejia-Wahba) |
| `lalonde_cps_obs.csv` | Comparación observacional | 15 992 × 11 | 0,71 MB | `e009f600a50b` | `causaldata` (CPS-1) |
| `retencion_observacional.csv` | Negocio (simulado) | 5 000 × 4 | 0,29 MB | `180f9e9c56ce` | DGP con semilla (REPLICACION Sección 5) |

Ninguno supera 25 MB, por lo que los CSV completos quedan en la carpeta (no se requiere muestra).
Los SHA256 completos están en `data/CHECKSUMS.txt`.

---

## 1) `card_krueger_1994.csv` — Salario mínimo, comida rápida NJ/PA (DiD)

- **Paper:** Card, D. & Krueger, A. B. (1994). *Minimum Wages and Employment: A Case Study of the
 Fast-Food Industry in New Jersey and Pennsylvania*. American Economic Review 84(4): 772–793.
- **Fuente primaria:** `https://davidcard.berkeley.edu/data_sets/njmin.zip` (archivo `public.dat` + `codebook`). Descarga directa verificada (HTTP 200, ZIP SHA256 `41a29aec…`, 410 observaciones).
- **Licencia/procedencia:** datos de investigación públicos publicados por David Card en su web
 institucional (Berkeley) como material de replicación del artículo.
- **Diseño:** dos olas de encuestas a **410 restaurantes** (Burger King, KFC, Roy Rogers, Wendy's).
 Ola 1 (feb–mar 1992, antes del alza del salario mínimo de NJ a 5,05 USD) y Ola 2 (nov–dic 1992, después). El empleo se mide como **FTE = tiempo completo + gerentes + 0,5 × medio tiempo**.
- **Distribución por estado:** `STATE=1` (New Jersey) = 331; `STATE=0` (Pennsylvania) = 79.

### Columnas clave (46 originales del codebook + 4 derivadas)

| Columna | Significado de negocio |
|---|---|
| `SHEET` | Identificador único de la tienda |
| `CHAIN` | Cadena: 1=Burger King, 2=KFC, 3=Roy Rogers, 4=Wendy's |
| `CO_OWNED` | 1 si la tienda es propiedad de la compañía (no franquicia) |
| `STATE` / `NJ` | 1 = New Jersey (tratado), 0 = Pennsylvania (control) |
| `EMPFT`, `EMPPT`, `NMGRS` | Empleados de tiempo completo, medio tiempo y gerentes — **ola 1 (antes)** |
| `EMPFT2`, `EMPPT2`, `NMGRS2` | Los mismos — **ola 2 (después)** |
| `WAGE_ST`, `WAGE_ST2` | Salario inicial por hora (antes / después) |
| `STATUS2` | Estado de la tienda en ola 2 (3 = cerrada) |
| `FTE`, `FTE2` (derivadas) | Empleo equivalente a tiempo completo, ola 1 / ola 2 |
| `DEMP` (derivada) | Cambio de empleo FTE = `FTE2 − FTE` |

> Resto de columnas (dummies de zona `SOUTHJ/CENTRALJ/NORTHJ/PA1/PA2/SHORE`, precios `PSODA/PFRY/PENTREE`,> horas `HRSOPEN`, cajas `NREGS`, etc.) conservan su nombre original del codebook. Faltantes = `NaN`
> (en `public.dat` venían como `.`).

### Benchmark de replicación (calculado con el venv el 19/07/2026)

- Cambio medio de FTE (medias por celda, como la tabla 2×2 del notebook y la hoja `tabla_did`):
 **NJ = +0,59**, **PA = −2,16** → **DiD = +0,59 − (−2,16) = +2,7536 FTE** (operativo del venv).
- La **regresión** `fte ~ nj + post + nj:post` sobre el **panel balanceado** (384 tiendas con ambas olas, SE clúster) da el mismo efecto: coef. `nj:post` = **+2,750** (SE 1,339); el DiD por regresión
 puede diferir del de medias unas décimas por el balanceo. *Etiqueta:* **operativo = venv**;
 **benchmark publicado (Card & Krueger 1994, Tabla 3)** ≈ **+2,76 (SE 1,36)** [otro pipeline] — el
 alza del salario mínimo en NJ **no** redujo el empleo.

---

## 2) `lalonde_nsw.csv` — NSW experimental (Dehejia-Wahba) (PSM)

- **Papers:** Rosenbaum, P. & Rubin, D. (1983). *The Central Role of the Propensity Score*.
 Biometrika 70(1). / LaLonde, R. (1986) y Dehejia & Wahba (1999, 2002).
- **Fuente:** paquete `causaldata` 0.1.4 (`nsw_mixtape`), muestra Dehejia-Wahba del experimento
 aleatorizado National Supported Work (NSW).
- **Diseño:** ensayo **aleatorizado** — 185 tratados (programa de empleo subsidiado) + 260 control = **445**.
- **Licencia/procedencia:** datos públicos de replicación redistribuidos por `causaldata`
 (companion de *The Effect*, Huntington-Klein) y por R. Dehejia (NBER).

### Columnas

| Columna | Significado de negocio |
|---|---|
| `data_id` | Origen de la muestra (`"Dehejia-Wahba Sample"`) |
| `treat` | 1 = participó en el programa (tratado), 0 = control |
| `age` | Edad (años) |
| `educ` | Años de escolaridad |
| `black` | 1 si afroamericano |
| `hisp` | 1 si hispano |
| `marr` | 1 si casado |
| `nodegree` | 1 si no completó secundaria |
| `re74`, `re75` | Ingreso real 1974 / 1975 (USD, pre-tratamiento) |
| `re78` | Ingreso real 1978 (USD) — **variable de resultado** |

### Benchmark (calculado con el venv el 19/07/2026)

- **ATT experimental** (diferencia de medias de `re78`, tratado − control) = **+1 794,34 USD**.
 *Etiqueta:* **benchmark experimental de referencia** (el "verdadero" contra el que se compara el ajuste observacional). Coincide con el valor tentativo declarado para la sesión (≈ +1 794).

---

## 3) `lalonde_cps_obs.csv` — Comparación observacional CPS-1

- **Fuente:** `causaldata` (`cps_mixtape`) — 15 992 individuos de la Current Population Survey (CPS-1),
 todos `treat=0` (grupo de comparación **no experimental**).
- **Esquema:** idénticas 11 columnas que `lalonde_nsw.csv`.
- **Uso:** la muestra **observacional** clásica de Dehejia-Wahba se arma al combinar los **185 tratados**
 de `lalonde_nsw.csv` con los **15 992 controles** de `lalonde_cps_obs.csv` (el notebook/QA hace la unión; no se materializa un tercer archivo para evitar duplicación).

### Benchmark (calculado con el venv el 19/07/2026)

- **Estimador naive observacional** (media `re78` tratados NSW − media `re78` controles CPS) =
 **−8 497,52 USD**. *Etiqueta:* **benchmark del sesgo** — el estimador ingenuo es fuertemente
 negativo y contradice al experimento; el objetivo pedagógico es que **PSM/ajuste por propensity
 score** recupere un ATT cercano al experimental (+1 794).

> **Nota de diseño (representación de los datos).** El paper de Card & Krueger es un panel **pareado
> por tienda** (misma tienda antes/después): el CSV llega en formato **ancho** (una fila por tienda con> columnas de ola 1 y ola 2), no en formato largo de dos grupos. El notebook debe calcular el DiD sobre
> `DEMP` (o pasar a formato largo antes de un `PanelOLS`). Para LaLonde, el experimental y el observacional
> vienen en **dos archivos separados**; la muestra observacional NO existe como un único CSV: se construye
> al unir tratados NSW + controles CPS. Ambos puntos deben manejarse explícitamente en el cuaderno.

---

## 4) `retencion_observacional.csv` — Campaña de retención (SIMULADO)

> **Dueño único del DGP: la ficha de la sesión de réplica del paper** (reconciliado 19/07/2026).
> Los parámetros son EXACTAMENTE los de esa sección; `descargar_datos.py` y el cuaderno los generan
> con el mismo código (el cuaderno lo hace **inline**, autocontenido).

- **Escenario de negocio:** una empresa de suscripción ofrece una **campaña de retención**. Los datos son
 **observacionales** (no aleatorizados): el equipo de marketing eligió a quién contactar a partir de la
 **actividad previa** del cliente (`engagement`) — los más activos fueron los más contactados.
 `engagement` también predice el **valor futuro**, por lo que es un **confusor** que sesga el estimador
 naive **al alza**; `tenure` (antigüedad) afecta al valor pero **no** a la asignación (covariable de precisión).
- **Generación:** `generar_retencion(n=5000, seed=20260719, true_ate=8.0)` con
 `value = 50 + 8·campaign + 20·engagement + 3·tenure + N(0, 8)`.

### Columnas

| Columna | Significado de negocio |
|---|---|
| `engagement` | Actividad previa del cliente (z-score) — **confusor** (engagement → campaign y → value) |
| `tenure` | Antigüedad en meses — **covariable de precisión** (tenure → value, no → campaign) |
| `campaign` | 1 = recibió la campaña de retención (**TRATO**) |
| `value` | Valor incremental del cliente en USD, próximo trimestre (**RESULTADO**) |

### Comportamiento observado (venv, 19/07/2026)

| Estimador | Valor | Lectura |
|---|---|---|
| Efecto **verdadero** del DGP (ATE = ATT, USD) | **+8,00** | constante de diseño (ancla dura) |
| Estimador **naive** (diferencia de medias) | **+23,64** | sesgado **al alza** por el confusor `engagement` |
| **Ajustado** (backdoor {engagement}, DoWhy) | **+7,95** | recupera el efecto verdadero |
| **Refutación** placebo / random common cause | **≈ 0 / ≈ 7,95** | el efecto se anula con tratamiento falso y resiste un confusor aleatorio |

*Etiqueta:* valores **operativos = venv**; los targets con tolerancia los fija `REPLICACION_PAPER.md` Secciones 4-5
(dueño único) y los audita el material de referencia de la sesión (hoja `causal_did`, celdas B6/B7/B8).

---

## Notas de verificación

- **Smoke de instalabilidad (19/07/2026):** `causaldata` 0.1.4, `dowhy` 0.14, `econml` 0.16.0,
 `linearmodels` 7.0 — todos importan y ejecutan en el venv (Python 3.13.5). No se requirió fallback.
- **Viabilidad de la réplica:** confirmada. Card & Krueger reproduce **+2,75** exacto; NSW reproduce el
 ATT experimental **+1 794,34** y el sesgo naive **−8 497,52**. No hay contradicción con los valores
 tentativos de lo declarado para la sesión
- **Regeneración:** `"<venv python>" "…/Sesiones/S12_inferencia_causal/data/descargar_datos.py"`.
