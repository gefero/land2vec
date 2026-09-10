"""Chequeo de uniformidad perceptual de la paleta de procesos del visor de clusters.

Reproduce -- sin dependencias -- la métrica de https://color-analyzer.streamlit.app/
(repo gefero/factor_data_uniform_tests): interpola la paleta a N muestras y calcula
el ΔE euclídeo entre muestras adyacentes; reporta media / σ / min / max. Cuanto
más baja y pareja la σ, más uniforme (viridis ≈ media 0.9, σ 0.3; Jet ≈ 2.0 / 0.85).

Acá se evalúa, para cada `proceso` de scripts/build_cluster_map.py:

  - la rampa temporal `process_color(key, t)` para t en [0, 1] (N pasos), en
    ΔE76 (CIELab, como la opción "CIE Lab" de la herramienta);
  - además el ΔE76 mínimo entre los colores base de todos los procesos
    (distinguibilidad del eje de proceso -- querés que sea alto).

Uso:
    python scripts/check_cluster_palette.py            # N = 64
    python scripts/check_cluster_palette.py --n 256
    python scripts/check_cluster_palette.py --hex      # imprime las rampas en hex
                                                       # (para pegar en el sitio)
"""

import argparse
import math

from build_cluster_map import (
    CONST_COLORS, PROCESSES, STATE_LABELS, _srgb_to_linear, process_color,
)

# --------------------------------------------------------------------------- #
#  sRGB -> CIELab (D65), Python puro
# --------------------------------------------------------------------------- #
_XN, _YN, _ZN = 0.95047, 1.0, 1.08883


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _lab(h: str) -> tuple[float, float, float]:
    r, g, b = (_srgb_to_linear(c) for c in _hex_to_rgb(h))
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / _XN
    y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / _YN
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / _ZN
    f = lambda t: t ** (1 / 3) if t > 216 / 24389 else (841 / 108) * t + 4 / 29
    fx, fy, fz = f(x), f(y), f(z)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def _de76(a: tuple, b: tuple) -> float:
    return math.dist(a, b)


def _stats(vals: list[float]) -> tuple[float, float, float, float]:
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    return m, sd, min(vals), max(vals)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=64, help="muestras por rampa (default 64)")
    ap.add_argument("--hex", action="store_true", help="imprimir cada rampa como lista de hex")
    args = ap.parse_args()

    print(f"ΔE76 (CIELab) entre pasos adyacentes -- {args.n} muestras por rampa\n")
    print(f"  {'proceso':<22} {'media':>7} {'σ':>7} {'min':>7} {'max':>7}")
    worst_sd = 0.0
    for key in PROCESSES:
        ramp = [process_color(key, i / (args.n - 1)) for i in range(args.n)]
        labs = [_lab(h) for h in ramp]
        des = [_de76(labs[i], labs[i + 1]) for i in range(len(labs) - 1)]
        m, sd, lo, hi = _stats(des)
        worst_sd = max(worst_sd, sd)
        print(f"  {key:<22} {m:7.3f} {sd:7.3f} {lo:7.3f} {hi:7.3f}")
        if args.hex:
            step = max(1, args.n // 7)
            print("        " + " ".join(ramp[::step]))

    # distinguibilidad entre procesos: ΔE76 del par más cercano de colores base
    bases = {k: _lab(process_color(k, 0.5)) for k in PROCESSES}
    keys = list(bases)
    pairs = [
        (_de76(bases[a], bases[b]), a, b)
        for i, a in enumerate(keys) for b in keys[i + 1:]
    ]
    pairs.sort()
    print(f"\nΔE76 entre colores base de procesos (t=0.5): min {pairs[0][0]:.1f}"
          f"  ({pairs[0][1]} ↔ {pairs[0][2]})")
    print("  5 pares más cercanos:")
    for de, a, b in pairs[:5]:
        print(f"    {de:6.1f}  {a} ↔ {b}")

    print(f"\nresumen: peor σ de rampa = {worst_sd:.3f}  "
          f"(objetivo < ~0.5; viridis ≈ 0.3)")

    # ---- paleta de estados constantes (fondo) ----
    print("\n" + "=" * 52)
    print("Paleta de estados constantes (fondo, pálida a propósito)\n")
    clabs = {t: _lab(h) for t, h in CONST_COLORS.items()}
    for t, hx in CONST_COLORS.items():
        print(f"  {t:<6} {hx}  {STATE_LABELS.get(t, '')}")
    cpairs = sorted(
        (_de76(clabs[a], clabs[b]), a, b)
        for i, a in enumerate(CONST_COLORS) for b in list(CONST_COLORS)[i + 1:]
    )
    print(f"\n  ΔE76 min entre estados: {cpairs[0][0]:.1f} ({cpairs[0][1]} ↔ {cpairs[0][2]})")
    print("  5 pares más cercanos:")
    for de, a, b in cpairs[:5]:
        print(f"    {de:6.1f}  {a} ↔ {b}")
    # rima de hue con el proceso análogo: mismo hue, el fondo mucho más claro (ΔL alto)
    rhyme = [("F", "regeneracion_bosque"), ("Wa", "dinamica_hidrica"),
             ("U", "urbanizacion"), ("A", "expansion_agricola")]
    print("\n  fondo vs proceso análogo (debe separarse en L):")
    for tok, proc in rhyme:
        cl, pl = clabs[tok], _lab(process_color(proc, 0.5))
        print(f"    {tok:<4} vs {proc:<20} ΔE76 {_de76(cl, pl):5.1f}  ΔL {cl[0]-pl[0]:+5.1f}")


if __name__ == "__main__":
    main()
