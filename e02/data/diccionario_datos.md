# Diccionario de datos — Sesión EPE E2 (Regresión lineal para predecir)

> Fecha de verificación de fuentes: **06/08/2026**. Datos obtenidos y validados
> programáticamente con `descargar_datos.py` (columnas clave + nº de filas). Reglas:
> español impersonal, fechas DD/MM/YYYY, UTF-8. Ningún dato es sintético.
>
> Ambos datasets ya fueron verificados para este curso —mismo esquema, mismo
> número de filas—; en local, el script busca primero una copia ya verificada.

---

## 1. `Advertising.csv` — Dataset de REGRESIÓN SIMPLE — REÚSO de S03

- **Rol:** caso de negocio de la regresión simple (¿cuánto mueve la inversión de un canal a las ventas?).
- **Contexto de negocio:** un anunciante registra, en **200 mercados**, el presupuesto de
 publicidad en tres canales y las ventas del producto. Dataset clásico de ISLR.
- **Procedencia canónica (sílabo EPE):** `https://www.statlearning.com/s/Advertising.csv`.
 Fallback verificado: mirror ISLR `nguyen-toan/ISLR` (mismo esquema y nº de filas). En local,
 `descargar_datos.py` **copia** la copia ya verificada de S03.
- **Licencia:** dataset educativo abierto del libro ISLR.
- **Tamaño:** 200 filas × 5 columnas (incluye una columna índice sin nombre) — ≈4 KB (< 25 MB → se conserva).

| Columna | Tipo | Significado de negocio |
|---|---|---|
| `Unnamed: 0` | entero | Índice de fila (no es variable de análisis). |
| `TV` | cuantitativa | Presupuesto de publicidad en **TV**, en **miles de USD**. |
| `radio` | cuantitativa | Presupuesto de publicidad en **radio**, en miles de USD. |
| `newspaper` | cuantitativa | Presupuesto de publicidad en **prensa**, en miles de USD. |
| `sales` | cuantitativa | **Ventas** del producto, en **miles de unidades** (variable respuesta `Y`). |

> Uso en la sesión: recta `ventas ~ TV` (interpretar la pendiente como unidades vendidas por> cada mil USD; leer el R²), comparación con el canal radio, y evaluación train/test (RMSE, MAE).

---

## 2. `AmesHousing_kaggle.csv` — Dataset de REGRESIÓN MÚLTIPLE — REÚSO de S04

- **Rol:** caso de negocio del laboratorio (predecir el precio de una vivienda a partir de sus características).
- **Contexto:** casas vendidas en **Ames, Iowa (2006–2010)**. Se usa el subconjunto de negocio
 (train de la competencia Kaggle «House Prices»), con nombres de columna sin espacios.
- **Procedencia canónica (sílabo EPE):** Kaggle «House Prices - Advanced Regression Techniques»
 o `sklearn.datasets.fetch_openml(data_id=42165)` (OpenML espeja el train de Kaggle). En local,
 `descargar_datos.py` **copia** la copia ya verificada de S04.
- **Origen del dataset:** De Cock, D. (2011), *Ames, Iowa: Alternative to the Boston Housing Data*,
 Journal of Statistics Education 19(3), DOI 10.1080/10691898.2011.11889627.
- **Licencia:** uso educativo/competencia; acceso abierto vía OpenML.
- **Tamaño:** 1 460 filas × 81 columnas — ≈1 MB (< 25 MB → se conserva). En la sesión se usan
 8 columnas; el resto queda disponible para exploración.

| Columna (usada en la sesión) | Tipo | Significado de negocio |
|---|---|---|
| `Id` | entero | Identificador de la vivienda (no es variable de análisis). |
| `SalePrice` | cuantitativa | **Precio de venta** en USD (variable respuesta `Y`). |
| `GrLivArea` | cuantitativa | **Superficie habitable** sobre rasante, en pie². |
| `OverallQual` | ordinal (1–10) | **Calidad general** de materiales y acabados. |
| `GarageCars` | discreta | Capacidad del garaje en **nº de plazas**. |
| `TotalBsmtSF` | cuantitativa | Superficie total del **sótano**, en pie². |
| `YearBuilt` | entero (año) | **Año de construcción**. |
| `Neighborhood` | nominal | **Barrio** (se usa para la dummy «premium»: NridgHt/NoRidge/StoneBr). |
| `GarageArea` | cuantitativa | Superficie del garaje, en pie² (solo para ilustrar la **redundancia** con `GarageCars`). |

> Uso en la sesión: modelo múltiple `SalePrice ~ GrLivArea + OverallQual + GarageCars +
> TotalBsmtSF + YearBuilt` (coeficientes parciales en USD y en % con `log`), dummy de barrio
> premium, idea de multicolinealidad (`GarageCars` vs `GarageArea`) y evaluación train/test. Se
> excluyen las **4 casas con `GrLivArea` > 4 000 pie²** (ventas atípicas que De Cock recomienda remover).

---

## 3. Notas de verificación y reproducibilidad

- **Idempotencia:** `descargar_datos.py` no re-obtiene un archivo si ya existe y valida su
 esquema (columnas clave + nº de filas). En local, **copia** desde `Sesiones/S03_ols/data/` y
 `Sesiones/S04_regresion_multiple/data/` para no re-descargar.
- **Colab:** el cuaderno incluye un respaldo portable — descarga Advertising del sitio de ISLR y
 Ames vía `fetch_openml(42165)` si la carpeta `data/` no está presente.
- **Trazabilidad de cifras:** toda cifra del material existe en el cuaderno ejecutado o en
 `resultados/E02_resultados.xlsx`.
