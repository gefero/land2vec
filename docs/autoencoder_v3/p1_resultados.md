# P1: compresión de las trayectorias (resultados)

**Fecha:** 2026-10-03. Ejecuta plan §4.1. Código: `scripts/modelo/p1_compresion.py`. Salidas: `data/autoencoder_v3/p1_metricas.csv`, `p1_entropia.csv`, `data/autoencoder_v3/p1/om_trate.npy` (distancia OM 4.554², 83 MB, no versionar) y `models/autoencoder_v3/p1/` (z, reconstrucciones y pesos por corrida).

## Montaje
- Universo: las 4.554 trayectorias de `universo_argentina.csv`. Se ajusta y se evalúa sobre el mismo universo (es descriptivo, sin partición).
- Peso de ajuste: min(n_px, 100), para que las 9 constantes (95 % de la superficie) no dominen. Las métricas se reportan ponderadas por superficie (`px`) y por tipo (`tipo`).
- Autoencoder: `TrajectoryAutoencoder`, n_embd 64, 2 capas, pooling por query, sin dropout, AdamW lr 1e-3 con warmup y coseno, `clip_grad_norm` 1,0, 300 épocas, lote 256. d ∈ {1,2,3,4,6,8,12,16} × 3 semillas.
- Lineales sin entrenar (sin semillas, mismos pesos): PCA del one-hot y MCA ponderado; se reconstruye tomando el argmax por año.
- Autoencoder lineal (`lin`, agregado el 2026-10-05): encoder y decoder lineales sobre el one-hot, con la misma pérdida (entropía cruzada por año) y los mismos pesos que el AE; Adam lr 1e-2, 3.000 pasos, 3 semillas. Es la comparación justa.
- Referencia: entropía de la distribución de trayectorias (bits): ponderada por superficie 2,97 (7,87 sólo dinámicas); con tope 10,90; uniforme sobre los tipos 12,15.

## Resultados (medias entre semillas; subconjunto: dinámicas)

| d | AE exacta (px) | AE exacta (tipo) | AE lineal exacta (tipo) | AE estados ok (tipo) | AE lineal estados ok (tipo) | AE err. año (tipo) | AE lineal err. año (tipo) | PCA exacta (tipo) | MCA exacta (tipo) |
|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0,106 | 0,018 | 0,001 | 0,151 | 0,069 | 2,19 | 3,55 | 0,000 | 0,000 |
| 2 | 0,852 | 0,258 | 0,005 | 0,400 | 0,078 | 0,34 | 2,34 | 0,000 | 0,000 |
| 3 | 0,994 | 0,485 | 0,019 | 0,616 | 0,152 | 0,16 | 1,72 | 0,000 | 0,001 |
| 4 | 0,997 | 0,641 | 0,048 | 0,756 | 0,226 | 0,11 | 1,16 | 0,000 | 0,001 |
| 6 | 0,999 | 0,815 | 0,213 | 0,889 | 0,425 | 0,05 | 0,49 | 0,000 | 0,002 |
| 8 | 1,000 | 0,880 | 0,422 | 0,931 | 0,632 | 0,03 | 0,28 | 0,009 | 0,001 |
| 12 | 1,000 | 0,944 | 0,740 | 0,975 | 0,861 | 0,02 | 0,09 | 0,061 | 0,051 |
| 16 | 1,000 | 0,957 | 0,893 | 0,979 | 0,947 | 0,01 | 0,03 | 0,165 | 0,132 |

Ponderado por superficie (dinámicas, exacta): AE lineal 0,11 / 0,29 / 0,80 / 0,96 / 0,998 con d = 3 / 4 / 6 / 8 / 12, contra 0,99 / 1,00 con el AE en d = 3 / 4.

Exactitud por año (tipo, dinámicas): AE 0,71 / 0,87 / 0,93 / 0,96 / 0,98 / 0,99 / 0,997 / 0,997 para d = 1…16; PCA 0,30 → 0,91; MCA 0,31 → 0,89. Macro F1 (9 clases) y recall de clases raras (Wa, B, Sp) siguen el mismo orden (`p1_metricas.csv`).

## Lectura
1. **El universo se comprime muy bien con el autoencoder.** Ponderado por superficie, con d = 3 se reconstruye exacta el 99,4 % de los píxeles dinámicos; con d = 2, el 85 %. Es coherente con la entropía por superficie (7,9 bits para las dinámicas): la superficie se concentra en pocas trayectorias (36 cubren el 50 %).
2. **Por tipo hay que subir d.** Las trayectorias raras necesitan d ≈ 8 (88 % exactas, 93 % con la secuencia de estados correcta, error de 0,03 años en el momento del cambio) y la ganancia a d = 12–16 es chica. Lo que se pierde primero al bajar d es el año exacto del cambio, después el orden de los estados en las trayectorias de 2–3 cambios.
3. **Contra lo lineal.** PCA y MCA con argmax son una línea de base débil: reconstruyen exacta una fracción mínima hasta d = 12 (≤ 6 %). El autoencoder lineal entrenado con la misma pérdida es mucho mejor (exactitud por año de 0,94 con d = 8 y 0,99 con d = 16, por tipo), así que **buena parte de la ventaja inicial del AE no lineal era del decodificador**. La ventaja real de la no linealidad se concentra en d bajo: a d = 4 el AE reconstruye exacto el 64 % de los tipos y el lineal el 5 %; a d = 8, 88 % contra 42 %; recién a d = 16 se acercan (96 % contra 89 %). En el error del año del cambio, el AE es de 5 a 10 veces menor hasta d = 12. Por superficie, el lineal alcanza lo del AE a partir de d = 8–12 (0,96 y 0,998), contra d = 3 del AE. Lectura: el AE comprime con unas 2 a 4 dimensiones menos que lo lineal a igual fidelidad.
4. **Ruido entre réplicas:** desvío entre semillas ≤ 0,02 en exactitud por tipo para d ≥ 2 (mayor en d = 1–3). Las diferencias AE vs. lineal lo superan por mucho; entre d consecutivos altos (12 vs 16), no.
5. **Estructura: el AE no preserva la métrica OM.** Solapamiento de vecinos (k = 10, tipo): AE 0,60 y casi plano desde d = 4; AE lineal, PCA y MCA suben hasta 0,72–0,74 con d = 16 (a d ≤ 6 el AE gana). Spearman con OM (tipo): AE 0,45–0,64, lineales (incluido el AE lineal) ≈ 0,66–0,77; ponderado por superficie es inestable (desvío entre semillas hasta 0,36) porque los pares quedan dominados por constantes. Esto es relevante para P2: la tipología en z agrupará distinto que OM.
6. **Costura:** excluir las trayectorias marcadas casi no mueve nada (exacta por tipo, dinámicas: 0,962 vs 0,957 con d = 16; 0,892 vs 0,880 con d = 8). El AE no tiene problema para reconstruir la costura. Los modelos no se reajustaron sin las marcadas: sólo se recalculan las métricas sobre el subconjunto.

## Elección de d y límites
- **No se elige una d.** P1 mide cuánta fidelidad da cada d y sube de forma monótona; no define por sí solo una d "mejor". Decisión del 2026-10-05: llevar varias d a P2 (d = 3, 8 y 16: el mínimo que cubre la superficie, un intermedio y el extremo de la grilla) y ver si la tipología depende de d.
- Límites: sin partición de validación (la pregunta es de compresión del universo, no de generalización); n_embd 64 y 300 épocas, distinto del `autoencoder_v2` (128); un solo peso de ajuste (tope 100) y una sola configuración, sin tunear; falta la comparación contra la entropía como cota por d.
