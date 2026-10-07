"""P1 (plan autoencoder_v3 §4.1): ¿cuánta información de las 4.554 trayectorias de Argentina
sobrevive a una compresión de d dimensiones? Autoencoder frente a PCA y MCA del one-hot.

Trabaja sobre data/autoencoder_v3/universo_argentina.csv (una fila por trayectoria distinta,
`n_px` = superficie). Es descriptivo: se ajusta y se evalúa sobre el mismo universo, no hay
partición de validación. Los datos de ajuste se ponderan con min(n_px, tope) para que las
9 constantes (95 % de la superficie) no dominen la pérdida.

Subcomandos (desde la raíz del repo):
    python scripts/modelo/p1_compresion.py om                       # distancia OM, cacheada en data/autoencoder_v3/p1/
    python scripts/modelo/p1_compresion.py linear                   # PCA y MCA para todas las d
    python scripts/modelo/p1_compresion.py train --d 8 --seed 0     # una corrida del autoencoder
    python scripts/modelo/p1_compresion.py linae                    # autoencoder lineal (misma pérdida que el AE)
    python scripts/modelo/p1_compresion.py eval                     # métricas de todo lo que haya -> p1_metricas.csv, p1_entropia.csv
"""
import argparse
import json
import time
from pathlib import Path
import sys as _sys
_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)
from land2vec import paths as P  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from land2vec.tokenizer import Tokenizer  # noqa: E402

DIMS = list(range(1, 32, 3))   # 1, 4, 7, ..., 31 (con 31 años de serie puede hacer falta un d mayor que en 2000-2022)
SEEDS = [0, 1, 2]
V = len(Tokenizer.VOCAB)  # 11 (incluye [UNK]=0)
UNIVERSO = P.DATA / "autoencoder_v3" / "universo_argentina.csv"
OUT_DATA = P.DATA / "autoencoder_v3" / "p1"
OUT_MODELS = P.MODELS_V3 / "p1"
TOPE = 100


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def load_universe() -> tuple[pd.DataFrame, np.ndarray]:
    u = pd.read_csv(UNIVERSO)
    X = np.array([Tokenizer.encode(s) for s in u.seqs], dtype=np.int64)
    assert (X > 0).all(), "tokens desconocidos"
    return u, X


def train_weights(u: pd.DataFrame, tope: int = TOPE) -> np.ndarray:
    return np.minimum(u.n_px.values, tope).astype(np.float64)


# ---------------------------------------------------------------------------
# Modelos lineales (ponderados): PCA y MCA del one-hot
# ---------------------------------------------------------------------------
def onehot(X: np.ndarray) -> np.ndarray:
    n, T = X.shape
    Z = np.zeros((n, T, V))
    Z[np.arange(n)[:, None], np.arange(T)[None, :], X] = 1.0
    return Z.reshape(n, T * V)


def _argmax_by_year(S: np.ndarray, T: int) -> np.ndarray:
    "S: (n, T*V) puntajes -> (n, T) tokens; el [UNK] (0) nunca se elige."
    S = S.reshape(len(S), T, V).copy()
    S[:, :, 0] = -np.inf
    return S.argmax(-1)


def pca_fit(X: np.ndarray, w: np.ndarray, dmax: int):
    Z = onehot(X)
    mu = (w[:, None] * Z).sum(0) / w.sum()
    A = np.sqrt(w)[:, None] * (Z - mu)
    _, s, Vt = np.linalg.svd(A, full_matrices=False)
    return {"mu": mu, "Vt": Vt[:dmax], "Z": Z}


def pca_embed_recon(m, d: int):
    Vd = m["Vt"][:d]
    z = (m["Z"] - m["mu"]) @ Vd.T
    return z, _argmax_by_year(m["mu"] + z @ Vd, m["Z"].shape[1] // V)


def mca_fit(X: np.ndarray, w: np.ndarray, dmax: int):
    Z = onehot(X)
    T = X.shape[1]
    keep = Z.sum(0) > 0
    Zk = Z[:, keep]
    W = w.sum()
    r = w / W
    c = (w[:, None] * Zk).sum(0) / (W * T)
    Pm = Zk * w[:, None] / (W * T)
    S = (Pm - np.outer(r, c)) / np.sqrt(r)[:, None] / np.sqrt(c)[None, :]
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    return {"r": r, "c": c, "U": U[:, :dmax], "s": s[:dmax], "Vt": Vt[:dmax], "keep": keep, "n": len(X)}


def mca_embed_recon(m, d: int):
    r, c, U, s, Vt, keep = m["r"], m["c"], m["U"][:, :d], m["s"][:d], m["Vt"][:d], m["keep"]
    z = U * s / np.sqrt(r)[:, None]  # coordenadas principales de las filas
    Ph = np.outer(r, c) + np.sqrt(r)[:, None] * ((U * s) @ Vt) * np.sqrt(c)[None, :]
    full = np.full((m["n"], len(keep)), -np.inf)
    full[:, keep] = Ph
    return z, _argmax_by_year(full, len(keep) // V)


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------
def train_ae(X: np.ndarray, w: np.ndarray, d: int, seed: int, epochs: int, batch: int, lr: float,
             n_embd: int, n_layer: int, pooling: str, verbose: bool = True, device: str = "cpu"):
    import torch
    from torch.nn import functional as F
    from land2vec.model import TrajectoryAutoencoder

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xt = torch.tensor(X).to(device)
    wt = torch.tensor(w / w.mean(), dtype=torch.float32).to(device)
    model = TrajectoryAutoencoder(V, X.shape[1], d, n_embd=n_embd, n_layer=n_layer, dropout=0.0, pooling=pooling).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps_ep = int(np.ceil(len(X) / batch))
    total, warm = epochs * steps_ep, 20 * steps_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(X))
        tot = torch.zeros((), device=device)   # se acumula en el dispositivo: sin sincronizar en cada paso
        for i in range(0, len(X), batch):
            idx = torch.tensor(perm[i:i + batch], device=device)
            logits = model(Xt[idx])
            ce = F.cross_entropy(logits.reshape(-1, V), Xt[idx].reshape(-1), reduction="none").view(len(idx), -1).mean(1)
            loss = (ce * wt[idx]).sum() / wt[idx].sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.detach() * len(idx)
        hist.append(tot.item() / len(X))
        if verbose and (ep % 25 == 0 or ep == epochs - 1):
            print(f"  d={d} s={seed} ep {ep:>3d} loss={hist[-1]:.5f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    zs, recs = [], []
    with torch.no_grad():
        for i in range(0, len(Xt), 512):   # por bloques: codificar todo de una vez pedía cientos de MB extra y daba OOM con varios procesos en la GPU
            zi = model.encode(Xt[i:i + 512])
            logits = model.decode(zi)
            logits[:, :, 0] = -float("inf")
            zs.append(zi.cpu())
            recs.append(logits.argmax(-1).cpu())
    return model, torch.cat(zs).numpy(), torch.cat(recs).numpy(), hist


def train_linear_ae(X: np.ndarray, w: np.ndarray, d: int, seed: int, steps: int, lr: float):
    """Autoencoder lineal sobre el one-hot, con la misma pérdida (entropía cruzada por año, mismos
    pesos) que el AE no lineal: z = W_e x + b_e, logits = W_d z + b_d. Es la comparación justa
    contra PCA/MCA, que reconstruyen con argmax sin optimizar una pérdida categórica."""
    import torch
    from torch.nn import functional as F
    torch.manual_seed(seed)
    n, T = X.shape
    Z = torch.tensor(onehot(X), dtype=torch.float32)
    Xt = torch.tensor(X)
    wt = torch.tensor(w / w.sum(), dtype=torch.float32)
    enc, dec = torch.nn.Linear(T * V, d), torch.nn.Linear(d, T * V)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.01)
    hist = []
    for st in range(steps):
        logits = dec(enc(Z)).view(n, T, V)
        ce = F.cross_entropy(logits.reshape(-1, V), Xt.reshape(-1), reduction="none").view(n, T).mean(1)
        loss = (ce * wt).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        hist.append(loss.item())
        if st % 500 == 0 or st == steps - 1:
            print(f"  lin d={d} s={seed} step {st} loss={hist[-1]:.5f}", flush=True)
    with torch.no_grad():
        z = enc(Z)
        logits = dec(z).view(n, T, V)
        logits[:, :, 0] = -float("inf")
        rec = logits.argmax(-1).numpy()
    return z.numpy(), rec, hist


# ---------------------------------------------------------------------------
# Métricas de fidelidad
# ---------------------------------------------------------------------------
def runs_of(row):
    "Estados sucesivos (sin años) y posiciones donde cambia."
    ch = np.flatnonzero(row[1:] != row[:-1]) + 1
    return tuple(row[np.r_[0, ch]]), ch


def per_traj(X: np.ndarray, R: np.ndarray) -> pd.DataFrame:
    n = len(X)
    same = X == R
    rows = []
    for i in range(n):
        st_x, ch_x = runs_of(X[i])
        st_r, ch_r = runs_of(R[i])
        ok_states = st_x == st_r
        rows.append((len(ch_x) == len(ch_r), ok_states,
                     np.abs(ch_x - ch_r).mean() if ok_states and len(ch_x) else np.nan,
                     len(ch_r) - len(ch_x)))
    a = np.array(rows, dtype=float)
    return pd.DataFrame({"acc_anio": same.mean(1), "exacta": same.all(1).astype(float),
                         "n_cambios_ok": a[:, 0], "estados_ok": a[:, 1], "err_anio_cambio": a[:, 2],
                         "dif_cambios": a[:, 3]})


def wmean(v, w):
    m = ~np.isnan(v)
    return float((v[m] * w[m]).sum() / w[m].sum()) if m.any() and w[m].sum() > 0 else np.nan


def subsets(u: pd.DataFrame) -> dict[str, np.ndarray]:
    din, sc = ~u.constante.values, ~u.costura.values
    return {"todas": np.ones(len(u), bool), "dinamicas": din, "constantes": ~din,
            "todas_sin_costura": sc, "dinamicas_sin_costura": din & sc}


def fidelity(u: pd.DataFrame, X: np.ndarray, R: np.ndarray) -> list[dict]:
    from sklearn.metrics import f1_score, recall_score
    pt = per_traj(X, R)
    out = []
    labels = sorted(set(np.unique(X)))
    for sname, mask in subsets(u).items():
        for wname, w in (("px", u.n_px.values.astype(float)), ("tipo", np.ones(len(u)))):
            ws = w[mask]
            for col in ("acc_anio", "exacta", "n_cambios_ok", "estados_ok", "err_anio_cambio"):
                out.append({"subset": sname, "peso": wname, "metrica": col, "valor": wmean(pt[col].values[mask], ws)})
            yt, yp = X[mask].ravel(), R[mask].ravel()
            sw = np.repeat(ws, X.shape[1])
            out.append({"subset": sname, "peso": wname, "metrica": "macro_f1",
                        "valor": float(f1_score(yt, yp, labels=labels, average="macro", sample_weight=sw, zero_division=0))})
            if sname in ("dinamicas", "todas"):
                rec = recall_score(yt, yp, labels=labels, average=None, sample_weight=sw, zero_division=0)
                for lab, v in zip(labels, rec):
                    out.append({"subset": sname, "peso": wname, "metrica": f"recall_{Tokenizer.REVERSE_VOCAB[lab]}", "valor": float(v)})
    return out


def structure(z: np.ndarray, D: np.ndarray, u: pd.DataFrame, k: int = 10, npairs: int = 2_000_000, seed: int = 0) -> list[dict]:
    "Correlación de Spearman entre distancias en z y OM (pares muestreados) y solapamiento de kNN."
    from scipy.spatial.distance import cdist
    from scipy.stats import spearmanr
    n = len(z)
    rng = np.random.default_rng(seed)
    Dz = cdist(z, z).astype(np.float32)
    out = []
    px = u.n_px.values.astype(float)
    for wname, p in (("tipo", None), ("px", px / px.sum())):
        i = rng.choice(n, npairs, p=p)
        j = rng.choice(n, npairs, p=p)
        m = i != j
        out.append({"peso": wname, "metrica": "spearman_om", "valor": float(spearmanr(Dz[i[m], j[m]], D[i[m], j[m]])[0])})
    nn_z = np.argsort(Dz + np.diag(np.full(n, np.inf, np.float32)), axis=1)[:, :k]
    nn_o = np.argsort(D + np.diag(np.full(n, np.inf)), axis=1)[:, :k]
    ov = np.array([len(set(a) & set(b)) / k for a, b in zip(nn_z, nn_o)])
    out.append({"peso": "tipo", "metrica": f"knn{k}_om", "valor": float(ov.mean())})
    out.append({"peso": "px", "metrica": f"knn{k}_om", "valor": float((ov * px).sum() / px.sum())})
    return out


def entropies(u: pd.DataFrame, w_train: np.ndarray) -> list[dict]:
    def H(w):
        p = w[w > 0] / w.sum()
        return float(-(p * np.log2(p)).sum())
    din = ~u.constante.values
    return [{"distribucion": "px_todas", "bits": H(u.n_px.values.astype(float))},
            {"distribucion": "px_dinamicas", "bits": H(u.n_px.values[din].astype(float))},
            {"distribucion": f"tope{TOPE}_todas", "bits": H(w_train)},
            {"distribucion": "uniforme_tipos", "bits": float(np.log2(len(u)))}]


# ---------------------------------------------------------------------------
# Subcomandos
# ---------------------------------------------------------------------------
def cmd_om(args):
    from land2vec import seqdist
    u, X = load_universe()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    path = OUT_DATA / "om_trate.npy"
    t0 = time.time()
    D, info = seqdist.build_distance(X, u.n_px.values.astype(float), method="om", sub="trate")
    np.save(path, D.astype(np.float32))
    print(f"OM {D.shape} en {time.time() - t0:.0f}s, {info} -> {P.rel(path)}")


def cmd_linear(args):
    u, X = load_universe()
    w = train_weights(u, args.tope)
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    for name, fit, emb in (("pca", pca_fit, pca_embed_recon), ("mca", mca_fit, mca_embed_recon)):
        m = fit(X, w, max(DIMS))
        for d in DIMS:
            z, R = emb(m, d)
            np.savez_compressed(OUT_MODELS / f"{name}_d{d}.npz", z=z.astype(np.float32), recon=R.astype(np.uint8))
        print(name, "listo")


def cmd_train(args):
    import torch
    torch.set_num_threads(args.threads)
    u, X = load_universe()
    w = train_weights(u, args.tope)
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    print(f"dispositivo: {device}", flush=True)
    model, z, R, hist = train_ae(X, w, args.d, args.seed, args.epochs, args.batch, args.lr,
                                 args.n_embd, args.n_layer, args.pooling, device=device)
    tag = f"ae_d{args.d}_s{args.seed}"
    np.savez_compressed(OUT_MODELS / f"{tag}.npz", z=z.astype(np.float32), recon=R.astype(np.uint8), loss=np.array(hist))
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, OUT_MODELS / f"{tag}.pt")
    (OUT_MODELS / f"{tag}.json").write_text(json.dumps(vars(args) | {"final_loss": hist[-1]}, indent=1, default=str))
    print(f"{tag}: loss final {hist[-1]:.5f}, acc año {(R == X).mean():.4f}")


def cmd_linae(args):
    import torch
    torch.set_num_threads(args.threads)
    u, X = load_universe()
    w = train_weights(u, args.tope)
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        for d in DIMS:
            z, R, hist = train_linear_ae(X, w, d, seed, args.steps, args.lr)
            np.savez_compressed(OUT_MODELS / f"lin_d{d}_s{seed}.npz", z=z.astype(np.float32),
                                recon=R.astype(np.uint8), loss=np.array(hist))
            print(f"lin_d{d}_s{seed}: acc año {(R == X).mean():.4f}", flush=True)


def cmd_eval(args):
    u, X = load_universe()
    om_path = args.om or OUT_DATA / "om_trate.npy"
    D = np.load(om_path) if om_path.exists() and om_path.stat().st_size > 1000 else None  # un puntero LFS pesa ~130 B
    if D is None:
        print("(sin OM cacheada: se omite la preservación de estructura; correr `om` primero)")
    rows = []
    for f in sorted(OUT_MODELS.glob("*.npz")):
        method, d, *seed = f.stem.split("_")
        d = int(d[1:])
        seed = int(seed[0][1:]) if seed else 0
        a = np.load(f)
        meta = {"metodo": method, "d": d, "semilla": seed}
        for r in fidelity(u, X, a["recon"].astype(np.int64)):
            rows.append(meta | r)
        if D is not None:
            for r in structure(a["z"], D, u):
                rows.append(meta | {"subset": "todas"} | r)
        print(f.stem, flush=True)
    out = P.DATA / "autoencoder_v3" / "p1_metricas.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(entropies(u, train_weights(u, args.tope))).to_csv(P.DATA / "autoencoder_v3" / "p1_entropia.csv", index=False)
    print("->", P.rel(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("om", cmd_om), ("linear", cmd_linear), ("train", cmd_train), ("linae", cmd_linae), ("eval", cmd_eval)):
        sp = sub.add_parser(name)
        sp.set_defaults(fn=fn)
        sp.add_argument("--tope", type=int, default=TOPE, help="peso de ajuste = min(n_px, tope)")
        if name == "eval":
            sp.add_argument("--om", type=Path, default=None, help="distancia OM (default: data/autoencoder_v3/p1/om_trate.npy)")
        if name == "linae":
            sp.add_argument("--steps", type=int, default=3000)
            sp.add_argument("--lr", type=float, default=1e-2)
            sp.add_argument("--threads", type=int, default=4)
        if name == "train":
            sp.add_argument("--d", type=int, required=True)
            sp.add_argument("--seed", type=int, default=0)
            sp.add_argument("--epochs", type=int, default=400)
            sp.add_argument("--batch", type=int, default=128)
            sp.add_argument("--lr", type=float, default=1e-3)
            sp.add_argument("--n-embd", type=int, default=128)
            sp.add_argument("--n-layer", type=int, default=2)
            sp.add_argument("--pooling", choices=["mean", "query"], default="query")
            sp.add_argument("--threads", type=int, default=2)
            sp.add_argument("--device", default="auto", help="auto (cuda si hay), cpu, cuda, cuda:1, ...")
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
