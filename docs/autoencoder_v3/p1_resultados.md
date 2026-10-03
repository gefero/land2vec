# P1: compresión de las trayectorias (resultados)

**Fecha:** 2026-10-03. Ejecuta plan §4.1. Código: `scripts/modelo/p1_compresion.py`. Salidas: `data/autoencoder_v3/p1_metricas.csv`, `p1_entropia.csv`, `data/autoencoder_v3/p1/om_trate.npy` (distancia OM 4.554², 83 MB, no versionar) y `models/autoencoder_v3/p1/` (z, reconstrucciones y pesos por corrida).

## Montaje
- Universo: las 4.554 trayectorias de `universo_argentina.csv`. Se ajusta y se evalúa sobre el mismo universo (es descriptivo, sin partición).
- Peso de ajuste: min(n_px, 100), para que las 9 constantes (95 % de la superficie) no dominen. Las métricas se reportan ponderadas por superficie (`px`) y por tipo (`tipo`).
- Autoencoder: `TrajectoryAutoencoder`, n_embd 64, 2 capas, pooling por query, sin dropout, AdamW lr 1e-3 con warmup y coseno, `clip_grad_norm` 1,0, 300 épocas, lote 256. d ∈ {1,2,3,4,6,8,12,16} × 3 semillas.
- Lineales (sin semillas, mismos pesos): PCA del one-hot y MCA ponderado; se reconstruye tomando el argmax por año.
- Referencia: entropía de la distribución de trayectorias (bits): ponderada por superficie 2,97 (7,87 sólo dinámicas); con tope 10,90; uniforme sobre los tipos 12,15.

## Resultados (medias entre semillas; subconjunto: dinámicas)

| d | AE exacta (px) | AE exacta (tipo) | AE estados ok (tipo) | AE err. año cambio (tipo) | PCA exacta (tipo) | MCA exacta (tipo) |
|--:|---:|---:|---:|---:|---:|---:|
| 1 | 0,106 | 0,018 | 0,151 | 2,19 | 0,000 | 0,000 |
| 2 | 0,852 | 0,258 | 0,400 | 0,34 | 0,000 | 0,000 |
| 3 | 0,994 | 0,485 | 0,616 | 0,16 | 0,000 | 0,001 |
| 4 | 0,997 | 0,641 | 0,756 | 0,11 | 0,000 | 0,001 |
| 6 | 0,999 | 0,815 | 0,889 | 0,05 | 0,000 | 0,002 |
| 8 | 1,000 | 0,880 | 0,931 | 0,03 | 0,009 | 0,001 |
| 12 | 1,000 | 0,944 | 0,975 | 0,02 | 0,061 | 0,051 |
| 16 | 1,000 | 0,957 | 0,979 | 0,01 | 0,166 | 0,132 |

Exactitud por año (tipo, dinámicas): AE 0,71 / 0,87 / 0,93 / 0,96 / 0,98 / 0,99 / 0,997 / 0,997 para d = 1…16; PCA 0,30 → 0,91; MCA 0,31 → 0,89. Macro F1 (9 clases) y recall de clases raras (Wa, B, Sp) siguen el mismo orden (`p1_metricas.csv`).

## Lectura
1. **El universo se comprime muy bien con el autoencoder.** Ponderado por superficie, con d = 3 se reconstruye exacta el 99,4 % de los píxeles dinámicos; con d = 2, el 85 %. Es coherente con la entropía por superficie (7,9 bits para las dinámicas): la superficie se concentra en pocas trayectorias (36 cubren el 50 %).
2. **Por tipo hay que subir d.** Las trayectorias raras necesitan d ≈ 8 (88 % exactas, 93 % con la secuencia de estados correcta, error de 0,03 años en el momento del cambio) y la ganancia a d = 12–16 es chica. Lo que se pierde primero al bajar d es el año exacto del cambio, después el orden de los estados en las trayectorias de 2–3 cambios.
3. **Diferencia grande con lo lineal.** A igual d, PCA y MCA reconstruyen exacta una fracción mínima de las trayectorias hasta d = 12 (≤ 6 %), y recién con d = 16 se acercan al AE de d = 3 en exactitud por año. Cautela: el decodificador lineal con argmax no optimiza una pérdida categórica, así que es una línea de base débil; un autoencoder lineal entrenado con la misma pérdida sería la comparación más justa (pendiente).
4. **Ruido entre réplicas:** desvío entre semillas ≤ 0,02 en exactitud por tipo para d ≥ 2 (mayor en d = 1–3). Las diferencias AE vs. lineal lo superan por mucho; entre d consecutivos altos (12 vs 16), no.
5. **Estructura: el AE no preserva la métrica OM.** Solapamiento de vecinos (k = 10, tipo): AE 0,60 y casi plano desde d = 4; PCA/MCA suben hasta 0,74 con d = 16. Spearman con OM (tipo): AE 0,45–0,64, lineales ≈ 0,70–0,77; ponderado por superficie es inestable (desvío entre semillas hasta 0,36) porque los pares quedan dominados por constantes. Esto es relevante para P2: la tipología en z agrupará distinto que OM.
6. **Costura:** excluir las trayectorias marcadas casi no mueve nada (exacta por tipo, dinámicas: 0,962 vs 0,957 con d = 16; 0,892 vs 0,880 con d = 8). El AE no tiene problema para reconstruir la costura. Los modelos no se reajustaron sin las marcadas: sólo se recalculan las métricas sobre el subconjunto.

## Propuesta de d y límites
- Se propone **d = 8** como compromiso para P2 (d = 3 si sólo importa la superficie). La decisión es de ustedes.
- Límites: sin partición de validación (la pregunta es de compresión del universo, no de generalización); n_embd 64 y 300 épocas, distinto del `autoencoder_v2` (128); un solo peso de ajuste (tope 100) sin explorar; falta el autoencoder lineal entrenado y la comparación contra la entropía como cota por d.
