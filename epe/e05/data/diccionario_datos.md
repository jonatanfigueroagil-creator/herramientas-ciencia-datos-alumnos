# Diccionario de datos — Sesión EPE E5 (Predecir categorías: churn y fraude)

> Fecha de verificación de fuentes: **06/08/2026**. Datos obtenidos y validados
> programáticamente con `descargar_datos.py` (esquema + nº de filas + checksum SHA256).
> Reglas: español impersonal, fechas DD/MM/YYYY, UTF-8. Ningún dato es sintético.
>
> Ambos casos usan datasets ya verificados para este curso: **Telco Churn** y
> **Credit Card Fraud**. El churn se recalcula en vivo. El fraude 99:1 y sus
> métricas operativas provienen de `resultados/S10_resultados.xlsx` (dataset completo,> que **nunca** vive en OneDrive).

---

## 1. `telco_churn.csv` — Dataset de CHURN (fuga de clientes) — REÚSO de S09

- **Rol:** caso de negocio de clasificación binaria (predecir la fuga de un cliente).
- **Contexto:** muestra pública de **IBM** (Watson Analytics). Cada fila es un cliente de
 una empresa de telecomunicaciones; el objetivo `Churn` indica si se fue en el último mes.
- **Procedencia canónica:** Kaggle — `blastchar/telco-customer-churn` (archivo oficial `WA_Fn-UseC_-Telco-Customer-Churn.csv`).
- **Mirror abierto usado** (el mismo que S09; Kaggle pide login):
 `https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
 En local, `descargar_datos.py` **copia el CSV ya verificado de S09** para no re-descargar.
- **Licencia:** muestra pública de IBM para fines educativos.
- **Tamaño:** 7 043 filas × 21 columnas, ≈0,97 MB (< 25 MB → se conserva en la carpeta),
 **SHA256** `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`.
- **Balance del target:** `Churn = Yes` = 1 869 — `No` = 5 174 (≈ **26,5 %** de fuga).
- **Limpieza documentada (idéntica a S09):** `TotalCharges` llega como texto; **11 filas**
 (clientes con `tenure = 0`, altas del mes) traen un **espacio en blanco** en lugar de un
 número. Al convertir a numérico esos 11 pasan a NaN y se **eliminan** → el modelo trabaja
 con **7 032 filas**.

| Columna | Tipo | Significado de negocio |
|---|---|---|
| `customerID` | texto | Identificador único del cliente. No es variable de análisis. |
| `gender`, `SeniorCitizen`, `Partner`, `Dependents` | categóricas | Perfil demográfico del cliente. |
| `tenure` | entero | **Antigüedad** en meses. Predictor de retención más fuerte (a mayor antigüedad, menor fuga). |
| `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | categóricas | Servicios contratados. |
| `Contract` | categórica | Tipo de contrato: `Month-to-month`, `One year`, `Two year`. **Palanca de retención** clave. |
| `PaperlessBilling`, `PaymentMethod` | categóricas | Facturación y medio de pago. |
| `MonthlyCharges` | float | Cargo mensual (USD). |
| `TotalCharges` | float (texto→num) | Cargo total acumulado (USD). 11 blancos en `tenure = 0`. |
| `Churn` | binaria {Yes,No} | **Objetivo:** el cliente se fue (`Yes` = 1) o se quedó (`No` = 0). |

> Uso en la sesión: modelo de **regresión logística** de churn (probabilidad de fuga),
> lectura en **odds ratio** (factores de retención), **matriz de confusión**, **ROC/AUC** y
> elección del **umbral por costo** (FN = cliente perdido ≫ FP = llamada de retención).
> El notebook reproduce el resultado operativo de S09: **AUC de test = 0,8361**.

---

## 2. `creditcard_muestra.csv` — Dataset de FRAUDE (desbalance 99:1) — REÚSO de S10

- **Rol:** caso de negocio de clasificación con clases **severamente desbalanceadas**
 (detección de fraude), para ilustrar la paradoja de la accuracy, ROC-AUC vs PR-AUC y el
 efecto de **SMOTE**.
- **Origen académico:** Dal Pozzolo, A., Caelen, O., Johnson, R. A. & Bontempi, G. (2015).
 *Calibrating Probability with Undersampling for Unbalanced Classification*. IEEE SSCI.
 Colaboración Worldline – Machine Learning Group (MLG), Université Libre de Bruxelles (ULB).
- **Procedencia canónica:** Kaggle — `mlg-ulb/creditcardfraud` (requiere login).
- **Mirror abierto usado** (el mismo que S10): HuggingFace
 `David-Egea/Creditcard-fraud-detection` → `.../resolve/main/creditcard.csv` (contenido idéntico al CSV oficial de Kaggle).
- **Licencia:** Database Contents License (DbCL); uso libre con crédito a ULB/Worldline.
- **Esquema (completo):** **284 807** filas × 31 columnas — **492 fraudes** (`Class = 1`) =
 **0,1727 %** (ratio ≈ **99,83: 0,17**). 0 nulos. Las 28 columnas `V1…V28` son componentes
 **PCA anonimizadas** (confidencialidad), más `Time`, `Amount` y `Class`.
- **Peso:** completo ~**150,8 MB** (>25 MB) → **NUNCA en OneDrive**. Se descarga en runtime a
 un caché temporal fuera de OneDrive (`descargar_datos.py --full`). **SHA256 (fuente completa)**
 `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`.

| Variable | Tipo | Significado de negocio |
|---|---|---|
| `Time` | float (s) | Segundos desde la primera transacción del período (≈ 48 h). |
| `V1` … `V28` | float | 28 componentes PCA anonimizadas (no interpretables individualmente). |
| `Amount` | float | Importe de la transacción (0 – 25 691,16). Útil para el análisis por costo. |
| `Class` | binaria {0,1} | **Objetivo:** 0 = legítima, 1 = **fraude** (clase minoritaria). |

### Muestra de laboratorio — `creditcard_muestra.csv`

- **Diseño (idéntico a S10):** TODOS los **492 fraudes** + submuestra aleatoria de **1 350**
 transacciones legítimas (`random_state=42`), ordenada por `Time`.
- **Esquema:** 1 842 filas × 31 columnas — **0,972 MB** (≤1 MB, se versiona) —
 **SHA256** `2841263a95459c6f0205deda18a8fa1a42e6842e739e8599f998922f9d55e37e`.
- **ATENCIÓN (nota de honestidad):** el ratio de la muestra es **26,71 %** de fraude — **NO**
 es el 99:1 del original. Es una muestra para **mirar los datos** en local. El **desbalance
 real 99:1 (0,17 %)** y todas las métricas operativas del fraude (matriz de confusión, precisión, recall, ROC-AUC, PR-AUC, efecto SMOTE) se toman de los **resultados ya validados
 de S10** sobre el **dataset completo** (`resultados/S10_resultados.xlsx`), reutilizados en el
 cuaderno. **Ningún conteo de fraude se calcula sobre la muestra** (daría otro número).

> Uso en la sesión: leer los datos con la muestra; demostrar la **paradoja de la accuracy**
> (99,83 % sin detectar un fraude), **precisión/recall** de la clase rara, **ROC-AUC vs PR-AUC**
> y el **trade-off de SMOTE** (recall ↑, precisión ↓) con los conteos reales de S10.

---

## 3. Trazabilidad de verificación (06/08/2026)

| Dataset | Fuente usada | Filas × cols | Positivos | SHA256 | Estado |
|---|---|---|---|---|---|
| Telco Churn (churn) | Reúso S09 (copia local) / mirror `treselle-systems` | 7 043 × 21 | 1 869 (26,5 %) | `16320c9c…3055e91` (local) | Verificado ✓ |
| Credit Card Fraud (fraude) | Reúso S10 / mirror HF `David-Egea` | 284 807 × 31 | 492 (0,1727 %) | `76274b69…551a89` (fuente) | Verificado ✓ (mirror) |
| — muestra de laboratorio | derivada (492 fraudes + 1 350 legítimas, rs=42) | 1 842 × 31 | 492 (26,71 %) | `2841263a…d55e37e` (local) | Reusada de S10 ✓ |

- **Idempotencia:** `descargar_datos.py` no re-obtiene un archivo si ya existe con esquema/SHA
 correcto. El completo de fraude se cachea fuera de OneDrive (`%TEMP%/curso_upc_e05_creditcard/`).
- **Métricas operativas del fraude 99:1:** `Sesiones/S10_clasificadores_clasicos/resultados/S10_resultados.xlsx`
 (reutilizadas en `resultados/E05_resultados.xlsx`).
