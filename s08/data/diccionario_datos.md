# Diccionario de datos — S08 Reglas de asociación + market basket

> Fecha de verificación de fuentes: **18/07/2026**. Genera los archivos con
> `"C:/Users/Usuario/OneDrive/Cursos/Herramientas de Ciencias de Datos/python" descargar_datos.py`
> (idempotente; añade `--smoke` para el smoke de mlxtend). Dos conjuntos: uno de
> **replicación** (benchmark FIMI) y uno de **negocio** (market basket sobre retail).

---

## 1. T10I4D100K — replicación (benchmark FIMI) — `T10I4D100K.dat`

**Rol:** dataset de replicación del paper de la técnica (Apriori). Es el benchmark
canónico del repositorio **FIMI** (Frequent Itemset Mining Implementations).

**Procedencia y licencia.**
- Generado por el **IBM Quest Synthetic Data Generator** descrito en
 **Agrawal, R. & Srikant, R. (1994), «Fast Algorithms for Mining Association
 Rules», Proc. 20th VLDB** — el paper de la réplica de esta sesión.
- Datos **sintéticos**, benchmark académico de **dominio público**; usado en
 cientos de artículos de itemset mining para comparar algoritmos.
- Fuente del sílabo: `http://fimi.uantwerpen.be/data/` → **RESUELTO**. Devolvía
 HTTP-404 el 18/07/2026, pero el sitio volvió a responder por **HTTPS**
 (`https://fimi.uantwerpen.be/data/`); re-verificado en vivo el **24/08/2026**
 (200, sirve `T10I4D100K.dat` con el contenido canónico).
- **Mirrors de respaldo (contenido idéntico, verificado por SHA256).** El `.dat`
 de FIMI y los dos mirrors abiertos en GitHub usados por el material del curso
 — `narendrababu-um/t10i4d100kdatapac`, archivo `data-raw/T10I4D100K.csv`
 (el repo lo nombra `.csv`, pero su contenido es el `.dat` transaccional crudo separado por espacios) y `yiweig/Apriori`, archivo `T10I4D100K.dat` (el que usa
 el material de referencia de la sesión e la ficha de la sesión de réplica del paper como
 fallback) — son **byte-idénticos** al original de FIMI: mismo SHA256 los tres
 y la caché local `T10I4D100K.dat`.
 - URL raw (mirror citado por el validador): `https://raw.githubusercontent.com/yiweig/Apriori/master/T10I4D100K.dat`
 - **SHA256 (FIMI, ambos mirrors y archivo local — idéntico): `464d7b87aa7fdc0dc432ddad0493515569b2968b0f229ce39399c454a8c62e0c`**

**Nomenclatura del nombre.** `T10I4D100K` codifica los parámetros del generador:
- **T10** = longitud media de una transacción ≈ 10 ítems.
- **I4** = longitud media de los itemsets potencialmente frecuentes ≈ 4.
- **D100K** = **100 000** transacciones (D = «number of transactions»).

**Formato (transaccional CRUDO, NO tabular).** Una **línea por transacción**
(ticket/canasta). Cada línea son los **ítems** de esa transacción, como **enteros
separados por un espacio**, terminada con un espacio y salto de línea. **No** hay
cabecera, comas ni columnas fijas: cada fila tiene un largo distinto. No se puede
leer con `pd.read_csv` como matriz; se parsea línea a línea
(`parse_transacciones` en `descargar_datos.py`).

Ejemplo (primeras líneas reales):
```
25 52 164 240 274 328 368 448 538 561 630 687 730 775 825 834
39 120 124 205 401 581 704 814 825 834
35 249 674 712 733 759 854 950
```

**Esquema verificado (18/07/2026).**
| Propiedad | Valor observado |
|---|---|
| Nº de transacciones (líneas) | **100 000** |
| Nº de ítems distintos | **870** (identificadores en el rango 0–999; solo 870 aparecen) |
| Longitud media de transacción | **10.10** ítems (mín. 1, máx. 29) |
| Nº total de ítems (suma de largos) | 1 010 228 |
| Peso del archivo | ~3.8 MB (<25 MB) → **se CONSERVA** en la carpeta |

**Significado de negocio.** Cada entero es un **producto anónimo** (SKU) y cada
línea es una **canasta de compra** (todos los productos de un mismo ticket). Al
ser sintético no hay etiquetas legibles: se usa para **medir y comparar
algoritmos** (Apriori vs FP-Growth) de forma reproducible, no para leer reglas de
negocio (eso se hace con el dataset de retail, abajo).

**Cómo se replica el paper.** Con `min_support` fijo se cuenta el **nº de itemsets
frecuentes** y se verifica que **Apriori y FP-Growth devuelven exactamente los
mismos** (son algoritmos EXACTOS: solo difieren en eficiencia). Ver la nota de
valores observados al final.

---

## 2. Online Retail II — canastas de mercado (negocio) — `online_retail_baskets_muestra.csv`

**Rol:** caso de negocio (market basket real, productos con nombre legible).
**Se REUTILIZA la misma fuente verificada en S07** (UCI id 502); aquí se prepara
de forma distinta porque la unidad de análisis es la **factura**, no el cliente.

**Procedencia y licencia.**
- **Chen, D. (2019). Online Retail II. UCI Machine Learning Repository.**
 `https://doi.org/10.24432/C5CG6D`, UCI id 502, **Licencia CC BY 4.0**.
- El **completo** son ~1.07 M filas en un `.xlsx` de **~45.6 MB** (dos hojas: «Year 2009-2010» y «Year 2010-2011») → **NUNCA se guarda en OneDrive**. Se
 descarga el `.zip` de UCI **solo en runtime** a un temporal del sistema, se
 limpia y se conserva una **muestra ≤1 MB**.
 - URL zip: `https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip`
 - **SHA256 (zip): `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb`**

**Limpieza (idéntica en intención a la de S07, con Description).**
1. Quitar filas sin `Customer ID`.
2. Quitar **cancelaciones** (`Invoice` que empieza por `'C'`).
3. Quitar `Quantity <= 0` y `Price <= 0` (devoluciones / precios nulos).
4. Quitar `Description` vacía y normalizar espacios.
- Resultado: **1 067 371 → 805 549 filas** (36 969 facturas, 5 241 productos).

**Cómo se arma la canasta por ticket (one-hot por factura).** La **transacción =
lista de `Description` de una misma `Invoice`** (productos distintos de eseticket). Es exactamente lo que consume `mlxtend`:
1. `canastas = df.groupby("Invoice")["Description"].apply(lambda s: sorted(set(s))).tolist`
 → lista de listas (una por factura).
2. `TransactionEncoder.fit_transform(canastas)` → matriz **one-hot booleana**
 (filas = facturas, columnas = productos; `True` = el producto está en el ticket).
 Se usa **dtype bool** por memoria (matriz muy dispersa).
3. `apriori(onehot, min_support=..., use_colnames=True)` / `fpgrowth(...)` →
 itemsets frecuentes; `association_rules(...)` → reglas de cross-selling.
- La función `canastas_por_factura` de `descargar_datos.py` implementa el paso 1.

> **Diferencia con la muestra de S07.** La `online_retail_II_muestra.csv` de S07
> se muestreó **por cliente** (para RFM) y **no incluye `Description`**; por eso
> **no sirve** para market basket. Esta muestra S08 se genera **por factura** y
> conserva `Description` (el ítem legible que se necesita para leer reglas).

**Muestra guardada (`online_retail_baskets_muestra.csv`).**
- **800 facturas** elegidas al azar (`RandomState=42`) con **todas sus líneas**
 → cada factura es una canasta **completa** (no truncada).
- **16 979 filas × 4 columnas**, **0.951 MB** (≤1 MB).
- **SHA256: `a3745101de08f44a8fc67167e9075cb12427a776b53e7d0cdc503a78ddae2760`**

| Columna | Tipo | Significado de negocio |
|---|---|---|
| `Invoice` | texto (6 díg.) | Nº de **factura/ticket** = identificador de la **canasta**. Todas las filas con la misma `Invoice` son un mismo ticket. |
| `StockCode` | texto | Código interno del **producto** (SKU). |
| `Description` | texto | **Nombre del producto** (el ítem de la canasta; base de las reglas legibles, p. ej. «GREEN REGENCY TEACUP AND SAUCER»). |
| `Country` | texto | País del cliente (mayoría «United Kingdom»); permite segmentar canastas por mercado. |

**En Colab / para el alumno.** El notebook usa esta muestra o baja el completo
en runtime (nunca exige API key). El script es idempotente y funciona local y en
Colab; el `.xlsx` completo y el pickle limpio quedan en un temporal fuera de
OneDrive.

---

## Nota de VALORES OBSERVADOS (referencia para la ficha de la sesión de réplica del paper)

> Estos son valores **observados** con el venv el 18/07/2026 (smoke `--smoke`).
> **No son los targets**: el dueño único de los targets con tolerancia es
> la ficha de la sesión de réplica del paper (`investigador-tema`). Se dejan como
> insumo verificado; **no contradicen** ningún target (aún por fijar).

- **T10I4D100K, `min_support = 0.01`:** Apriori = **385** itemsets frecuentes y
 FP-Growth = **385** → **coinciden en conteo Y en conjunto exacto** (diferencia 0),
 como exige la teoría (ambos son exactos). Reparto por tamaño: 375 de 1 ítem,
 9 de 2 ítems, 1 de 3 ítems. Reglas con `confidence ≥ 0.5`: **7**
 (la más fuerte, `{39,704} → {825}`, conf 0.935, lift 30.3).
- **Online Retail II (muestra 800 facturas), `min_support = 0.02`:** Apriori = 215
 itemsets = FP-Growth (coinciden); **38** reglas con `confidence ≥ 0.3` (las de
 mayor lift son los juegos de tazas «Regency Teacup», lift ≈ 33 (cociente exacto del venv 32,727), cross-selling claro).
- **Viabilidad de la réplica: OK.** El formato transaccional público reproduce el
 diseño del benchmark FIMI (100 000 tickets, ítems enteros); `mlxtend` carga,
 hace one-hot bool y ejecuta Apriori/FP-Growth sin problemas. Los conteos son
 deterministas (no dependen de semilla) → aptos para tolerancia 0.

---

### Referencias cruzadas
- Script generador: `data/descargar_datos.py` (funciones `obtener_t10i4d100k`, `parse_transacciones`, `obtener_retail_baskets`, `canastas_por_factura`, `smoke`).
- Réplica y targets: la ficha de la sesión de réplica del paper (dueño de los targets).
- Preparación one-hot para el alumno: `plantillas/guia_one_hot_transacciones.md`.
- Catálogo global: el material de referencia de la sesión (fila S08 actualizada 18/07/2026).
