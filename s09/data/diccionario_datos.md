# Diccionario de datos — Sesión S09 (Regresión Logística + GLM)

> Curso "Herramientas para la Ciencia de Datos" — UPC. Tres conjuntos: dos de
> **replicación** (logística y GLM Poisson) anclados en el paper/libro seminal
> y uno de **negocio** (churn). Descarga y verificación reproducibles con
> `descargar_datos.py` (idempotente, local + Colab, con checksum SHA256).
> **Fecha de verificación de las fuentes: 18/07/2026.**

Los tres archivos pesan < 25 MB, por lo que se conservan **completos** en la
carpeta `data/` (ninguno requiere muestra). El notebook los descarga en runtime
con el mismo script.

---

## 1. SAheart — South African Heart Disease *(replicación logística, ESL cap. 4)*

- **Archivo local:** `SAheart.data` — 462 filas × 10 variables (+ `chd`).
- **Unidad de análisis:** un varón adulto de una región de alto riesgo
 coronario del Cabo Occidental (Sudáfrica). Estudio retrospectivo de
 cardiopatía coronaria (CHD).
- **Rol:** dataset con el que *The Elements of Statistical Learning* ilustra la
 regresión logística (Tabla 4.2: coeficientes en log-odds de `chd` sobre los factores de riesgo). Es la **réplica** de la sesión.
- **Procedencia y fallback:** enlace del sílabo
 `https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data` responde
 **HTTP-403** a scripts (bloqueo anti-bot). Fallback aplicado (idéntico contenido, con cabecera): mirror abierto en GitHub
 `empathy87/The-Elements-of-Statistical-Learning-Python-Notebooks`,
 `data/South African Heart Disease.txt`.
- **Cita:** Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of
 Statistical Learning*, 2.ª ed., Sección 4.4. Springer. Datos originales: Rousseauw,
 J. et al. (1983), *South African Medical Journal* 64, 430–436.
- **Licencia:** dominio público académico (dataset de libro de texto).
- **SHA256 (fuente):** `c158781284d05a24265bb3764e0c48ad9e0dc9f646b27b0b44d43eba46299805`

| Variable | Tipo | Significado (negocio / clínico) |
|---|---|---|
| `sbp` | numérica | Presión arterial sistólica (mmHg). Marcador de hipertensión. |
| `tobacco` | numérica | Tabaco acumulado consumido (kg a lo largo de la vida). |
| `ldl` | numérica | Colesterol LDL ("malo") en sangre. |
| `adiposity` | numérica | Índice de adiposidad (grasa corporal). |
| `famhist` | categórica {Present, Absent} | Antecedente **familiar** de cardiopatía. Se codifica dummy `Present=1`. |
| `typea` | numérica | Puntaje de conducta tipo A (estrés/competitividad). |
| `obesity` | numérica | Índice de obesidad (tipo IMC). |
| `alcohol` | numérica | Consumo actual de alcohol. |
| `age` | numérica | Edad del paciente (años). |
| `chd` | **binaria 0/1 (target)** | Presencia de cardiopatía coronaria. `1`=enfermo. |

- **Balance del target:** `chd` = 160 casos positivos / 462 (34,6 %).
 `famhist` Present = 192 / 462. (columna auxiliar `row.names` se descarta.)

---

## 2. ships — Averías en cascos de barcos *(replicación GLM Poisson)*

- **Archivo local:** `ships.csv` — 40 filas × 5 variables (+ índice).
- **Unidad de análisis:** una combinación tipo de barco × año de construcción ×
 periodo de operación, con la **exposición** agregada (`service`).
- **Rol:** ejemplo canónico de **regresión de Poisson con offset**
 `log(exposure)` de McCullagh & Nelder; sirve para conteo de eventos y
 diagnóstico de sobre-dispersión (deviance/df).
- **Procedencia:** Rdatasets (mirror de datasets de R), paquete `MASS`:
 `https://vincentarelbundock.github.io/Rdatasets/csv/MASS/ships.csv`.
- **Cita:** McCullagh, P. & Nelder, J. A. (1989). *Generalized Linear Models*,
 2.ª ed. Chapman & Hall. Paper de la familia GLM: Nelder, J. A. & Wedderburn,
 R. W. M. (1972), *JRSS-A* 135(3).
- **Licencia:** `MASS` (GPL-2/3), uso académico.
- **SHA256 (fuente):** `447f68b31465fb092dc542bdc7a0ddc0b80f5b983314b29b0667b25a11b5f115`

| Variable | Tipo | Significado (negocio) |
|---|---|---|
| `type` | categórica {A,B,C,D,E} | Tipo de barco de carga. Se codifica con dummies. |
| `year` | categórica (60/65/70/75) | Año de construcción del casco (cohorte de diseño). |
| `period` | categórica (60/75) | Periodo de operación observado. |
| `service` | numérica | **Exposición**: meses-buque agregados de servicio. Va como `offset = log(service)` (a más servicio, más ocasiones de avería). |
| `incidents` | **conteo ≥ 0 (target)** | Número de averías/incidentes en el casco. Es la respuesta Poisson. |

- **Notas de modelado:** 34 de las 40 filas tienen `service > 0` (las 6 con exposición 0 no aportan al offset y se excluyen). Total de `incidents` = 356.
 El cociente **deviance/df ≈ 1,55** observado indica **sobre-dispersión leve**
 (punto didáctico: motiva quasi-Poisson o binomial negativa).

---

## 3. Telco Customer Churn *(caso de negocio — churn)*

- **Archivo local:** `telco_churn.csv` — 7 043 filas × 21 columnas.
- **Unidad de análisis:** un cliente de una empresa de telecomunicaciones.
 Target `Churn` (se dio de baja o no).
- **Rol:** laboratorio de negocio de la sesión — modelo de **fuga de clientes**
 con regresión logística, interpretación en odds ratio y evaluación con matriz
 de confusión / ROC / AUC + elección de umbral según costo del falso negativo.
- **Procedencia y fallback:** enlace del sílabo Kaggle
 `blastchar/telco-customer-churn` **requiere login** (decisión del curso 17/07/2026: sin token de API). Fallback aplicado (mismo CSV oficial de IBM, `WA_Fn-UseC_-Telco-Customer-Churn.csv`): mirror abierto en GitHub
 `treselle-systems/customer_churn_analysis`. El alumno nunca necesita API key.
- **Cita:** IBM (2019). *Telco Customer Churn* (muestra de IBM Cognos / Watson Analytics). **Licencia:** muestra pública de IBM para fines educativos.
- **SHA256 (fuente):** `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

| Variable | Tipo | Significado (negocio) |
|---|---|---|
| `customerID` | id | Identificador del cliente. Se descarta antes de modelar. |
| `gender` | categórica | Género del cliente (Female/Male). |
| `SeniorCitizen` | binaria 0/1 | 1 si es adulto mayor. |
| `Partner` | categórica | Tiene pareja (Yes/No). |
| `Dependents` | categórica | Tiene dependientes a cargo (Yes/No). |
| `tenure` | numérica | **Antigüedad** en meses como cliente. Predictor clave de retención. |
| `PhoneService` | categórica | Contrató servicio de telefonía. |
| `MultipleLines` | categórica | Múltiples líneas (o "No phone service"). |
| `InternetService` | categórica | Tipo de internet (DSL / Fiber optic / No). |
| `OnlineSecurity` | categórica | Seguridad en línea contratada. |
| `OnlineBackup` | categórica | Respaldo en línea contratado. |
| `DeviceProtection` | categórica | Protección de dispositivo. |
| `TechSupport` | categórica | Soporte técnico premium. |
| `StreamingTV` | categórica | Servicio de TV por streaming. |
| `StreamingMovies` | categórica | Servicio de películas por streaming. |
| `Contract` | categórica | Tipo de contrato (Month-to-month / One year / Two year). Fuerte predictor de churn. |
| `PaperlessBilling` | categórica | Facturación electrónica (Yes/No). |
| `PaymentMethod` | categórica | Medio de pago (cheque, transferencia, tarjeta, débito automático). |
| `MonthlyCharges` | numérica | Cargo **mensual** al cliente. |
| `TotalCharges` | numérica* | Cargo **acumulado** total. *(ver limpieza abajo)*. |
| `Churn` | **binaria (target)** | El cliente se dio de baja (`Yes`) o sigue activo (`No`). |

- **Balance del target:** `Churn` = 1 869 Yes / 5 174 No (**26,5 %** de fuga en el crudo de 7 043; **26,6 %** tras limpiar los 11 NaN → 7 032, que es el dato del modelo).
- **Limpieza documentada de `TotalCharges`:** la columna llega como **texto**.
 Hay **11 filas** (clientes con `tenure = 0`, es decir altas del mismo mes)
 que traen un **espacio en blanco** en vez de un número. Al convertir a
 numérico esos 11 espacios pasan a **NaN**. Estrategia por defecto en el
 script (`limpiar_telco`): **eliminarlos** → el modelo queda con **7 032
 filas**; alternativa razonable: imputar 0 (cliente nuevo sin cargo acumulado).

---

## Viabilidad de la réplica — valores OBSERVADOS (18/07/2026)

> Estos son valores **observados** por el smoke `descargar_datos.py --smoke`, no
> targets. Los **targets cuantitativos con tolerancia los define
> la ficha de la sesión de réplica del paper** (dueño único: `investigador-tema`).
> Se anotan aquí solo como evidencia de que los datos permiten reproducir el
> resultado del paper.

- **(a) Logística SAheart** (`statsmodels.Logit`, **modelo de 7 predictores**
 `chd ~ sbp+tobacco+ldl+famhist+obesity+alcohol+age`, `famhist`
 dummy, n=462): `tobacco` coef **+0,07953** (OR 1,083), `famhist` coef
 **+0,93919** (OR 2,558), `age` coef **+0,04254** (OR 1,044), `ldl` coef
 **+0,18478** (OR 1,203); pseudo-R² McFadden **0,208**. Reproduce ESL
 Tabla 4.2 de forma exacta (tobacco 0,080, famhist 0,939, age 0,043, ldl 0,185).
- **(b) GLM Poisson ships** (`statsmodels.GLM(family=Poisson)`, offset
 `log(service)`, n=34): deviance **38,70** / df 25 → **deviance/df ≈ 1,55**
 (sobre-dispersión leve); coef `C(type)[T.E]` **+0,3256** (tasa ×1,385).
- **(c) Churn Telco** (`sklearn.LogisticRegression`, one-hot 30 features, train/test 80/20, `StandardScaler`): **AUC(test) = 0,836**; matriz de
 confusión al umbral 0,5 = `[[918, 115], [159, 215]]` (N_test = 1 407).
 Consistente con el ~0,84 tentativo declarado para la sesión.

**Nota de diseño para el notebook:** el CSV público de Telco representa
`SeniorCitizen` como 0/1 (no texto) y `TotalCharges` como texto con 11 blancos;
tratarlos como se indica arriba antes de ajustar. SAheart trae `famhist` como
texto {Present, Absent} → dummy. `ships` exige el **offset** `log(service)`; sin
él el coeficiente de Poisson no es una tasa comparable con McCullagh & Nelder.
