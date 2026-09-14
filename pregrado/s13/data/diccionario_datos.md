# Diccionario de datos — S13 Análisis de supervivencia

> Variables en español con su significado de negocio, procedencia, licencia y fecha de verificación. Los tres datasets se (re)generan con `descargar_datos.py` (idempotente, funciona local y en Colab, con checksum SHA256). Verificación real ejecutada el **06/08/2026** con el venv del curso (la matriz de versiones certificada del curso, `lifelines==0.30.3`, Python 3.13.14).

**Peso:** los tres archivos pesan < 25 MB (Telco 0,970 MB) → se conservan completos en el repo; no hay muestra separada.

---

## 1. `rossi_recidivism.csv` — REPLICACIÓN (paper seminal de Cox)

- **Qué es:** 432 ex-reclusos varones liberados de prisiones de Maryland, seguidos durante 1 año (52 semanas) en un experimento aleatorizado de ayuda económica post-liberación. Es el dataset didáctico clásico del **modelo de regresión de Cox** (Allison, *Survival Analysis Using SAS*; Fox & Weisberg).
- **Rol:** replicación de la regresión de Cox con `lifelines`.
- **Fuente primaria:** `lifelines.datasets.load_rossi` (integrado en lifelines; sin descarga de red). Fallback sin lifelines: Rdatasets, paquete R `carData`, dataset `Rossi` (subconjunto de las 9 columnas estándar).
- **Licencia/procedencia:** dataset de dominio en la literatura de supervivencia; redistribuido en el paquete `lifelines` (licencia MIT) y en `carData` (GPL-2/3). Uso académico libre.
- **Dimensiones verificadas:** 432 filas × 9 columnas; eventos (re-arresto) = 114/432 (26,4 %); 318 censurados (no re-arrestados al cierre del seguimiento).
- **SHA256 (archivo local):** `8e5f9d64821b5f98bb32fda57eb8466f1a40ed18b41c6d85dcc16a15988a1ddc`

| Columna | Tipo | Significado (negocio / estudio) |
|---|---|---|
| `week` | entero 1–52 | **Duración**: semana del re-arresto o de censura (fin del seguimiento). |
| `arrest` | 0/1 | **Evento**: 1 = re-arrestado durante el año; 0 = censurado (sin re-arresto). |
| `fin` | 0/1 | Recibió ayuda económica (tratamiento aleatorizado): 1 = sí, 0 = no. |
| `age` | entero | Edad en años al momento de la liberación. |
| `race` | 0/1 | 1 = negro, 0 = otro (codificación original del estudio). |
| `wexp` | 0/1 | Experiencia laboral previa a la prisión: 1 = sí. |
| `mar` | 0/1 | Estado civil: 1 = casado, 0 = no casado. |
| `paro` | 0/1 | Liberado bajo libertad condicional (parole): 1 = sí. |
| `prio` | entero | Número de condenas previas. |

- **Viabilidad de la réplica (verificada 06/08/2026):** Rossi SÍ permite reproducir la Cox con lifelines. Cox multivariante observado (log-HR): `fin` −0,3794 (HR 0,684), `prio` +0,0915, `age` −0,0574, `mar` −0,4337. Coincide con el resultado canónico de la documentación de lifelines y de Allison. *(Los targets con tolerancia los fija `investigador-tema` en la ficha de la sesión de réplica del paper; aquí solo se registran valores observados como evidencia de viabilidad, no como targets.)*

---

## 2. `veterans_lung_cancer.csv` — REPLICACIÓN 2 (Cox, 1972)

- **Qué es:** ensayo clínico aleatorizado de la Veterans' Administration (VA) con 137 pacientes de cáncer de pulmón avanzado; dos regímenes de quimioterapia. Evento = muerte. Es uno de los conjuntos con que se ilustró el modelo de Cox (Cox, 1972; Kalbfleisch & Prentice, *The Statistical Analysis of Failure Time Data*, 1980).
- **Rol:** segunda replicación del modelo de Cox (con tipos de tumor y estado funcional de Karnofsky).
- **Nota (error conocido, entorno.md):** `scikit-survival` **NO instala en Python 3.13** (dependencia `ecos` sin wheel), por lo que **NO se usa `sksurv`**. Veterans se obtiene del **mirror abierto Rdatasets** (paquete R `survival`, dataset `veteran`).
- **Fuente:** `https://vincentarelbundock.github.io/Rdatasets/csv/survival/veteran.csv` (HTTP 200 verificado 06/08/2026).
- **Licencia/procedencia:** paquete R `survival` (LGPL-2+); Rdatasets recopila datasets de dominio académico. Uso libre.
- **Dimensiones verificadas:** 137 filas × 9 columnas (incluye `rownames` como índice original); eventos (muerte) = 128/137 (93,4 %); 9 censurados.
- **SHA256 (fuente = archivo local):** `3fba5cb9b15a10ab94d54e28b54d2c39d95c27b0fcff1c57aee0eb7edfd93950`

| Columna | Tipo | Significado (negocio / estudio) |
|---|---|---|
| `rownames` | entero | Índice de fila original de R (no es una variable del modelo). |
| `trt` | 1/2 | Tratamiento: 1 = estándar, 2 = experimental (test). |
| `celltype` | texto | Tipo histológico del tumor: `squamous`, `smallcell`, `adeno`, `large`. |
| `time` | entero | **Duración**: días de supervivencia desde el inicio del tratamiento. |
| `status` | 0/1 | **Evento**: 1 = fallecido; 0 = censurado. |
| `karno` | 0–100 | Índice de Karnofsky (estado funcional): mayor = mejor estado general. |
| `diagtime` | entero | Meses desde el diagnóstico hasta el ingreso al estudio. |
| `age` | entero | Edad en años. |
| `prior` | 0/10 | Terapia previa: 0 = no, 10 = sí (codificación original). |

- **Nota de representación (para el notebook):** `celltype` es categórica de 4 niveles → requiere codificación *one-hot* con una categoría de referencia (p. ej. `squamous`) para la Cox. `trt` está codificado 1/2 (no 0/1). No hay estructura pareada: es un diseño de dos grupos con covariables.
- **Viabilidad (verificada 06/08/2026):** Cox observado — `karno` log-HR −0,0328 (HR 0,968; el mejor estado funcional reduce fuertemente el riesgo), `trt` no significativo. Consistente con Kalbfleisch & Prentice / Cox 1972. *(Targets: los define `investigador-tema`.)*

---

## 3. `telco_churn.csv` — NEGOCIO (laboratorio de churn)

- **Qué es:** 7043 clientes de una empresa de telecomunicaciones (dataset de ejemplo de IBM). `tenure` = meses de antigüedad = **tiempo hasta el evento**; `Churn = Yes` = evento (baja). Sustituto de negocio (no es réplica de paper).
- **Rol:** laboratorio de supervivencia aplicado a churn (curvas KM por segmento + Cox + CLV).
- **Reutilización:** es la **misma fuente ya verificada en S09** (`Sesiones/S09_logistica_glm/data/telco_churn.csv`); se reusa el mirror abierto byte-idéntico del CSV oficial de IBM. El SHA256 coincide con el registrado en S09.
- **Fuente:** Kaggle `blastchar/telco-customer-churn` requiere login → **mirror abierto**: `https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv` (HTTP 200 verificado 06/08/2026).
- **Licencia/procedencia:** *IBM Sample Data Set — Telco Customer Churn* (IBM Cognos / Watson Analytics), de uso educativo abierto; espejado públicamente en GitHub.
- **Dimensiones verificadas:** 7043 filas × 21 columnas; `Churn` = 1869 «Yes» / 5174 «No» (26,5 % de bajas).
- **SHA256 (fuente = archivo local):** `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

| Columna | Tipo | Significado (negocio) |
|---|---|---|
| `customerID` | texto | Identificador del cliente. |
| `gender` | texto | Género (`Male`/`Female`). |
| `SeniorCitizen` | 0/1 | Cliente de la tercera edad. |
| `Partner` | Yes/No | Tiene pareja. |
| `Dependents` | Yes/No | Tiene personas a cargo. |
| `tenure` | entero (meses) | **Duración**: meses de antigüedad como cliente (tiempo al evento). |
| `PhoneService` | Yes/No | Contrató servicio telefónico. |
| `MultipleLines` | texto | Líneas múltiples (o `No phone service`). |
| `InternetService` | texto | Tipo de internet: `DSL`, `Fiber optic`, `No`. |
| `OnlineSecurity` … `StreamingMovies` | texto | Servicios adicionales contratados (6 columnas de add-ons). |
| `Contract` | texto | Tipo de contrato: `Month-to-month`, `One year`, `Two year` (segmento clave para KM/log-rank). |
| `PaperlessBilling` | Yes/No | Facturación electrónica. |
| `PaymentMethod` | texto | Medio de pago. |
| `MonthlyCharges` | float | Cargo mensual (USD) — insumo del CLV. |
| `TotalCharges` | float (texto crudo) | Cargo acumulado (USD). **11 celdas en blanco** (clientes con `tenure=0`) → NaN. |
| `Churn` | Yes/No | **Evento**: `Yes` = el cliente se dio de baja; `No` = censurado (sigue activo). |

- **Nota de representación / preparación (para el notebook):** es un diseño de supervivencia con **censura por la derecha**: la duración es `tenure` y el evento es `Churn=Yes`; los clientes activos son observaciones censuradas. Limpieza mínima (implementada en `limpiar_telco`): `TotalCharges` a numérico → eliminar las 11 filas NaN (7043 → 7032). El segmento `Contract` produce curvas KM muy separadas (log-rank Month-to-month vs Two year: χ²≈1550, p≈0), lo que valida su uso didáctico.

---

## Papers seminales — estado

Los papers seminales de la técnica (Kaplan-Meier 1958 y Cox 1972) son de **acceso cerrado** (JSTOR / Wiley / Taylor & Francis) y anteriores a arXiv; no existe versión abierta con licencia de redistribución. Ver `papers/README_papers.md` para las citas y enlaces verificados marcados «no redistribuible». Su registro formal (cita + base original) corresponde a la ficha de la sesión de réplica del paper (dueño: `investigador-tema`).
