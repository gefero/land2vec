"""Arma el informe de la Pregunta 1 (docs/autoencoder_v3/p1_resultados.md): rellena la plantilla
docs/autoencoder_v3/plantillas/p1_resultados.plantilla.md con tablas calculadas de data/autoencoder_v3/pregunta1/*.csv.

    python scripts/viz/pregunta1_informe.py      # unos 4 minutos (el análisis de tramos de un año recorre todos los modelos)

La plantilla tiene el texto; cada {{T_<nombre>}} se reemplaza por la tabla T["<nombre>"]. Las figuras las genera
scripts/viz/pregunta1_figuras.py.
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "scripts/validacion"), str(R / "scripts/modelo"), str(R / "src")]
import evaluacion_pregunta1 as ev  # noqa: E402
import p1_compresion as p1  # noqa: E402

IN = R / "data/autoencoder_v3/pregunta1"
rec = pd.read_csv(IN / "p11_reconstruccion.csv")
par = pd.read_csv(IN / "p11_parametros.csv")
MET = ["AE", "AE lineal", "PCA", "MCA"]
DSEL = sorted(rec.d.unique())          # todas las d
ORDEN_H = ["1", "2", "3", "4-5", "6+"]


def f(x, nd=3):
    return "—" if pd.isna(x) else f"{x:.{nd}f}".replace(".", ",")


def fr(v, nd=3):
    v = pd.Series(v).dropna()
    if len(v) == 0:
        return "—"
    return f"{f(v.mean(), nd)} ({f(v.min(), nd)}–{f(v.max(), nd)})" if len(v) > 1 else f(v.iloc[0], nd)


def md(rows, header, first="---"):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join([first] + ["--:"] * (len(header) - 1)) + "|"]
    return "\n".join(out + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def tab_rec(metrica, peso, conj, A, ds=DSEL, nd=3):
    s = rec[(rec.conjunto == conj) & (rec.A == A) & (rec.B.isin(["todos", "-"])) & (rec.peso == peso) & (rec.metrica == metrica)]
    return md([[d] + [fr(s[(s.d == d) & (s.metodo == m)].valor, nd) for m in MET] for d in ds], ["d"] + MET)


def tab_grad(metrica, d):
    orden = ["1", "2", "3", "4-5", "6+"]
    s = rec[(rec.conjunto == "mundo no visto") & (rec.peso == "tipo") & (rec.metrica == metrica) & (rec.d == d)]
    rows = []
    for a in ("visto", "nuevo"):
        for b in orden:
            q = s[(s.A == a) & (s.B == b)]
            rows.append([f"{a}, h = {b}", f"{int(q.n_tipos.iloc[0]):,}".replace(",", ".")] + [fr(q[q.metodo == m].valor) for m in MET])
    return md(rows, ["estrato", "tipos"] + MET)


def tab_sec(d):
    rows = []
    for mt, nom, nd in (("n_cambios_ok", "número de cambios correcto", 3), ("err_anio_cambio", "error de fechado (años)", 3),
                        ("f1_macro", "F1 macro por clase", 3)):
        for conj, A, lab in (("Argentina (ajuste)", "-", "Argentina"), ("mundo no visto", "visto", "visto"), ("mundo no visto", "nuevo", "nuevo")):
            s = rec[(rec.conjunto == conj) & (rec.A == A) & (rec.B.isin(["todos", "-"])) & (rec.peso == "tipo") & (rec.metrica == mt) & (rec.d == d)]
            rows.append([nom, lab] + [fr(s[s.metodo == m].valor, nd) for m in MET])
    return md(rows, ["métrica", "conjunto"] + MET)


# exploratorio: tramos de 1 año
def tramos():
    nv = ev.cargar_no_vistos()
    X = nv["X"].astype(int)
    sp = lambda r: np.diff(np.r_[0, np.flatnonzero(r[1:] != r[:-1]) + 1, len(r)])  # noqa: E731
    t1 = np.array([(sp(r) == 1).any() for r in X])
    vis, B = nv["visto"], ev.estrato_B(nv["h"])
    u, Xa = p1.load_universe()
    din = ~u.constante.values
    t1a = np.array([(sp(r) == 1).any() for r in Xa])
    comp = []
    for a in ("visto", "nuevo"):
        for b in ["1", "2", "3", "4-5", "6+"]:
            mk = (vis == (a == "visto")) & (B == b)
            comp.append([f"{a}, h = {b}", f"{100 * t1[mk].mean():.0f} %"])
    comp.insert(0, ["Argentina (ajuste)", f"{100 * t1a[din].mean():.0f} %"])
    grupos = (("Argentina, sin tramo de 1 año", din & ~t1a, "a"), ("visto, h = 1, sin tramo de 1 año", vis & (B == "1") & ~t1, "w"),
              ("visto, h = 1, con tramo de 1 año", vis & (B == "1") & t1, "w"), ("visto, h = 2–3, sin tramo de 1 año", vis & np.isin(B, ["2", "3"]) & ~t1, "w"),
              ("visto, h ≥ 4, sin tramo de 1 año", vis & np.isin(B, ["4-5", "6+"]) & ~t1, "w"))
    rows = []
    for met, d in [(m, d) for m in ("ae", "lin", "pca", "mca") for d in DSEL]:
        vals = {g: [] for g, _, _ in grupos}
        for s in ((0, 1, 2) if met in ("ae", "lin") else (None,)):
            c = ev.cargar_codigos(met, d, s)
            pw, pa = p1.per_traj(X, c["R_mundo"].astype(int)), p1.per_traj(Xa, c["R_arg"].astype(int))
            for g, mk, w in grupos:
                vals[g].append((pa if w == "a" else pw).estados_ok.values[mk].mean())
        rows.append([f"{ev.NOMBRE[met]}, d = {d}"] + [fr(vals[g]) for g, _, _ in grupos])
    ns = [f"{int(mk.sum()):,}".replace(",", ".") for _, mk, _ in grupos]
    return md(comp, ["estrato", "trayectorias con algún tramo de 1 año"]), md(rows, ["método"] + [f"{g} ({n})" for (g, _, _), n in zip(grupos, ns)])


def tab_par():
    return md([[d] + [f"{int(par[(par.metodo == m) & (par.d == d)].parametros.iloc[0]):,}".replace(",", ".") for m in MET] for d in DSEL],
              ["d"] + MET)


# --- anexo: tablas completas ---
CONJ = (("Argentina (ajuste)", "-", "Argentina (ajuste)"), ("mundo no visto", "visto", "mundo no visto, proceso visto"),
        ("mundo no visto", "nuevo", "mundo no visto, proceso nuevo"))


def tab_grad_todas(metrica, A):
    s = rec[(rec.conjunto == "mundo no visto") & (rec.A == A) & (rec.peso == "tipo") & (rec.metrica == metrica)]
    n = [f"{int(s[s.B == b].n_tipos.iloc[0]):,}".replace(",", ".") for b in ORDEN_H]
    rows = [[m, d] + [fr(s[(s.metodo == m) & (s.d == d) & (s.B == b)].valor) for b in ORDEN_H] for m in MET for d in DSEL]
    return md(rows, ["método", "d"] + [f"h = {b} ({x} tipos)" for b, x in zip(ORDEN_H, n)])


def anexo_rec(metricas, pesos):
    out = []
    for mt, nom in metricas:
        for peso, pnom in pesos:
            for conj, A, cnom in CONJ:
                out.append(f"**{nom}, {cnom}, {pnom}**\n\n" + tab_rec(mt, peso, conj, A))
    return "\n\n".join(out)


T = {"rec_acc_arg": tab_rec("acc_anio", "tipo", "Argentina (ajuste)", "-"), "rec_acc_vis": tab_rec("acc_anio", "tipo", "mundo no visto", "visto"),
     "rec_acc_nue": tab_rec("acc_anio", "tipo", "mundo no visto", "nuevo"),
     "rec_est_arg": tab_rec("estados_ok", "tipo", "Argentina (ajuste)", "-"), "rec_est_vis": tab_rec("estados_ok", "tipo", "mundo no visto", "visto"),
     "rec_est_nue": tab_rec("estados_ok", "tipo", "mundo no visto", "nuevo"),
     "rec_ex_tipo_arg": tab_rec("exacta", "tipo", "Argentina (ajuste)", "-"), "rec_ex_tipo_vis": tab_rec("exacta", "tipo", "mundo no visto", "visto"),
     "rec_ex_tipo_nue": tab_rec("exacta", "tipo", "mundo no visto", "nuevo"),
     "rec_ex_sup_arg": tab_rec("exacta", "superficie", "Argentina (ajuste)", "-"), "rec_ex_sup_vis": tab_rec("exacta", "superficie", "mundo no visto", "visto"),
     "rec_ex_sup_nue": tab_rec("exacta", "superficie", "mundo no visto", "nuevo"),
     "grad_est_4": tab_grad("estados_ok", 4), "grad_acc_4": tab_grad("acc_anio", 4), "grad_est_31": tab_grad("estados_ok", 31),
     "sec_4": tab_sec(4), "sec_31": tab_sec(31), "par": tab_par(),
     "anx_grad": "\n\n".join(f"**{nom}, proceso {A}**\n\n" + tab_grad_todas(mt, A) for mt, nom in
                              (("estados_ok", "Secuencia de estados correcta"), ("acc_anio", "Exactitud por año"), ("exacta", "Reconstrucción exacta"))
                              for A in ("visto", "nuevo")),
     "anx_sup": anexo_rec((("acc_anio", "Exactitud por año"), ("estados_ok", "Secuencia de estados correcta")), (("superficie", "ponderada por superficie"),)),
     "anx_sec": anexo_rec((("n_cambios_ok", "Número de cambios correcto"), ("err_anio_cambio", "Error de fechado (años)"),
                           ("f1_macro", "F1 macro por clase")), (("tipo", "cada tipo pesa 1"), ("superficie", "ponderada por superficie")))}
T["tramos_comp"], T["tramos"] = tramos()

tpl = (R / "docs/autoencoder_v3/plantillas/p1_resultados.plantilla.md").read_text()
miss = set(re.findall(r"\{\{T_(\w+)\}\}", tpl)) - set(T)
assert not miss, miss
out = R / "docs/autoencoder_v3/p1_resultados.md"
out.write_text(re.sub(r"\{\{T_(\w+)\}\}", lambda mo: T[mo.group(1)], tpl))
print("escrito", out)
