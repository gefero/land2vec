"""Tuneo del autoencoder sobre América Latina (protocolo: docs/autoencoder_v3/tuneo.md).

Todo es interrumpible y se retoma con EL MISMO COMANDO. Para parar:
  - Ctrl+C (o SIGTERM): cada corrida guarda su checkpoint y sale;
  - crear el archivo models/autoencoder_v3/tuneo/STOP: igual, pero sin tocar la terminal (el próximo lanzamiento lo borra);
  - --max-horas H o --hasta HH:MM: a esa hora se para solo.
Un corte abrupto (apagado, kill -9) pierde a lo sumo lo hecho desde el último checkpoint (cada --ckpt-seg segundos).

Subcomandos, en orden (desde la raíz del repo):
    python scripts/modelo/tuneo_ae.py particion                    # partición 80/20 y etiquetas de procesos (§3)
    python scripts/modelo/tuneo_ae.py referencias                  # S de z al azar, one-hot y PCA d=8 (§4.4)
    python scripts/modelo/tuneo_ae.py piloto --workers 1           # configuración base, d=8, 100.000 pasos (§6.1)
    python scripts/modelo/tuneo_ae.py resumen                      # métricas; del piloto sale S_max
    python scripts/modelo/tuneo_ae.py busqueda --s-max 80000 --workers 3   # 24 configuraciones hasta S_max (§6.2)
    python scripts/modelo/tuneo_ae.py confirmar --workers 3        # 4 mejores x 3 semillas x d 4, 8, 16, 20 (§6.3)
    python scripts/modelo/tuneo_ae.py final --config c07 --workers 3       # configuración elegida, todo AL (§6.5)
    python scripts/modelo/tuneo_ae.py estado                       # avance de todas las corridas

Opciones de los lanzadores: --device (auto | cpu | cuda | cuda:0,cuda:1), --workers, --threads, --max-horas, --hasta,
--ckpt-seg. `correr --dir <carpeta>` entrena una sola corrida (lo usan los lanzadores).

Salidas: models/autoencoder_v3/tuneo/<etapa>/<corrida>/ (config.json, estado.json, ckpt.pt, metricas.jsonl,
modelo.pt, codigos.npz, log.txt) y data/autoencoder_v3/tuneo/ (particion.npz, referencias.csv, plan_*.json,
metricas_<etapa>.csv).
"""
import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT / "scripts" / "modelo"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from land2vec import paths as P  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from land2vec import criterio_procesos as C  # noqa: E402
from land2vec.procesos import PROCESOS  # noqa: E402
from land2vec.tokenizer import Tokenizer  # noqa: E402
from land2vec.universos import cargar  # noqa: E402

V = len(Tokenizer.VOCAB)
DATA = P.DATA / "autoencoder_v3" / "tuneo"
MODELS = P.MODELS_V3 / "tuneo"
STOP = MODELS / "STOP"
NODO = socket.gethostname()

BASE = {"tope": 100, "dropout": 0.1, "weight_decay": 1e-2, "n_embd": 128, "n_layer": 2, "pooling": "query",
        "lr": 1e-3, "batch": 128}
ESPACIO = {"tope": [1, 10, 100, None], "dropout": [0.0, 0.1, 0.2], "weight_decay": [0.0, 1e-2, 1e-1],
           "n_embd": [64, 128, 256], "n_layer": [1, 2, 4], "pooling": ["mean", "query"], "lr": [3e-4, 1e-3, 3e-3],
           "batch": [128, 512]}
N_BUSQUEDA = 24
D_BUSQUEDA = 8
D_CONFIRMAR = (4, 8, 16, 20)
N_CONFIRMAR = 4
D_FINAL = (2, 3, 4, 6, 8, 12, 16, 20, 24, 31)
SEMILLAS = (0, 1, 2)
PILOTO_PASOS = 100_000
EVAL_CADA = 5_000
SALIDA_INTERRUMPIDA = 3


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def escribir_json(f: Path, obj):
    "Escritura atómica: un corte a mitad de camino deja el archivo anterior intacto."
    tmp = f.with_suffix(f.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, f)


def leer_json(f: Path, defecto=None):
    return json.loads(f.read_text()) if f.exists() else defecto


def huella(X: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(X.astype(np.uint8)).tobytes()).hexdigest()


def datos():
    "Universo de AL, máscara de validación y etiquetas (de particion.npz)."
    u = cargar("latam")
    f = DATA / "particion.npz"
    assert f.exists(), f"falta {P.rel(f)}: correr `tuneo_ae.py particion`"
    z = np.load(f)
    assert str(z["huella"]) == huella(u.X), "particion.npz no corresponde al censo de AL actual: rehacer `particion`"
    et = {p: (z[f"pos_{p}"], z[f"anio_{p}"]) for p in PROCESOS}
    return u, z["val"], et


def pesos_ajuste(n_px: np.ndarray, tope) -> np.ndarray:
    return (n_px if tope is None else np.minimum(n_px, tope)).astype(np.float64)


def n_parametros(cfg: dict) -> int:
    from land2vec.model import TrajectoryAutoencoder
    m = TrajectoryAutoencoder(V, P.V3_YEARS[1] - P.V3_YEARS[0] + 1, cfg["d"], n_embd=cfg["n_embd"], n_head=4,
                              n_layer=cfg["n_layer"], dropout=0.0, pooling=cfg["pooling"])
    return int(sum(p.numel() for p in m.parameters()))


# ---------------------------------------------------------------------------
# Partición y referencias (§3, §4.4)
# ---------------------------------------------------------------------------
def cmd_particion(args):
    u = cargar("latam")
    et = C.etiquetas(u.X)
    val = C.particion(u.X, u.area_km2, et, semilla=0)
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA / "particion.npz", val=val, huella=huella(u.X),
                        **{f"pos_{p}": et[p][0] for p in PROCESOS}, **{f"anio_{p}": et[p][1] for p in PROCESOS})
    din = u.dinamica
    filas = [{"conjunto": n, "tipos": int(s.sum()), "km2": u.area_km2[s].sum()} | {p: int((et[p][0] & s).sum()) for p in PROCESOS}
             for n, s in (("entrenamiento", ~val), ("validacion", val), ("constantes (entrenamiento)", ~din))]
    print(pd.DataFrame(filas).to_string(index=False, float_format="{:,.0f}".format))
    print("->", P.rel(DATA / "particion.npz"))


def cmd_referencias(args):
    import p1_compresion as p1
    u, val, et = datos()
    filas = []
    rng = np.random.default_rng(0)
    oh = p1.onehot(u.X).astype(np.float32)
    ajuste = p1.pca_fit(u.X[~val], pesos_ajuste(u.n_px[~val], BASE["tope"]), D_BUSQUEDA)
    z_pca, _ = p1.pca_apply(ajuste, u.X, D_BUSQUEDA)
    for nombre, A, metrica, Z in (("azar (d=8)", rng.standard_normal((len(u), D_BUSQUEDA)), "euclidean", None),
                                  ("one-hot (Hamming)", u.X, "hamming", oh),
                                  ("PCA (d=8)", z_pca, "euclidean", None)):
        t0 = time.time()
        m = C.agrupamiento(A if Z is None else Z, et, val, u.area_km2)
        m |= C.sonda(C.vecinos(A[~val], A[val], metric=metrica), et, val, u.area_km2)
        filas.append({"referencia": nombre} | m)
        print(f"{nombre:20s} S={m['S']:.3f}  S_sup={m['S_sup']:.3f}  S_vecinos={m['S_vecinos']:.3f}  "
              + " ".join(f"k{k}={m[f'grupos{k}_S']:.2f}" for k in C.K_GRUPOS) + f"  ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(filas).to_csv(DATA / "referencias.csv", index=False)
    print("->", P.rel(DATA / "referencias.csv"))


# ---------------------------------------------------------------------------
# Una corrida (reanudable)
# ---------------------------------------------------------------------------
def lr_en(paso: int, cfg: dict) -> float:
    "Calentamiento lineal (5 % de s_max) y coseno hasta el 1 %; depende sólo del paso (§5)."
    calent = max(1, int(0.05 * cfg["s_max"]))
    if paso < calent:
        return cfg["lr"] * (paso + 1) / calent
    t = min(1.0, (paso - calent) / max(1, cfg["s_max"] - calent))
    return cfg["lr"] * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t)))


class Parada:
    "Pedido de parada ordenada: señal (Ctrl+C, SIGTERM) o archivo STOP."
    def __init__(self):
        self.pedida = False
        for s in (signal.SIGINT, signal.SIGTERM):
            signal.signal(s, self._senal)

    def _senal(self, *_):
        self.pedida = True

    def mirar(self) -> bool:
        self.pedida = self.pedida or STOP.exists()
        return self.pedida


def evaluar(model, u, val, et, cfg, device) -> tuple[dict, np.ndarray, np.ndarray]:
    import torch
    model.eval()
    Xt = torch.tensor(u.X)
    zs, rs = [], []
    with torch.no_grad():
        for i in range(0, len(Xt), 1024):
            z = model.encode(Xt[i:i + 1024].to(device))
            lg = model.decode(z)
            lg[:, :, 0] = -float("inf")
            zs.append(z.cpu())
            rs.append(lg.argmax(-1).cpu())
    model.train()
    z, R = torch.cat(zs).numpy(), torch.cat(rs).numpy().astype(np.uint8)
    if cfg["universo"] == "todo":   # modelos finales: sin validación, sólo reconstrucción
        return C.reconstruccion(u.X, R, np.ones(len(u), bool)), z, R
    m = C.agrupamiento(z, et, val, u.area_km2)                                  # criterio principal (§4.1)
    m |= C.sonda(C.vecinos(z[~val], z[val]), et, val, u.area_km2) | C.reconstruccion(u.X, R, val)
    m |= {f"{k}_entrenamiento": v for k, v in C.reconstruccion(u.X, R, ~val).items()}
    return m, z, R


def cmd_correr(args):
    import torch
    from torch.nn import functional as F
    from land2vec.model import TrajectoryAutoencoder

    d = Path(args.dir)
    cfg = leer_json(d / "config.json")
    objetivo = cfg["objetivo"]
    estado = leer_json(d / "estado.json", {})
    escribir_json(d / "estado.json", estado | {"estado": "en curso", "pid": os.getpid(), "nodo": NODO, "inicio": time.time(),
                                               "ckpt_seg": args.ckpt_seg})
    parada = Parada()
    torch.set_num_threads(args.threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)

    u, val, et = datos()
    idx_tr = np.flatnonzero(~val) if cfg["universo"] == "entrenamiento" else np.arange(len(u))
    w = pesos_ajuste(u.n_px, cfg["tope"])
    wt = torch.tensor(w / w[idx_tr].mean(), dtype=torch.float32, device=device)
    Xt = torch.tensor(u.X, device=device)

    torch.manual_seed(cfg["seed"])
    model = TrajectoryAutoencoder(V, u.X.shape[1], cfg["d"], n_embd=cfg["n_embd"], n_head=4, n_layer=cfg["n_layer"],
                                  dropout=cfg["dropout"], pooling=cfg["pooling"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    paso, ckpt = 0, d / "ckpt.pt"
    if ckpt.exists():
        st = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        paso = st["paso"]
        torch.set_rng_state(st["rng_cpu"])
        if device.startswith("cuda") and st.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(st["rng_cuda"], device=torch.device(device))
        print(f"retomando en el paso {paso:,} (objetivo {objetivo:,})", flush=True)
    paso_inicio = paso
    escribir_json(d / "estado.json", leer_json(d / "estado.json") | {"paso_inicio": paso_inicio, "paso": paso})

    def guardar():
        st = {"model": model.state_dict(), "opt": opt.state_dict(), "paso": paso, "rng_cpu": torch.get_rng_state(),
              "rng_cuda": torch.cuda.get_rng_state(torch.device(device)) if device.startswith("cuda") else None}
        torch.save(st, ckpt.with_suffix(".tmp"))
        os.replace(ckpt.with_suffix(".tmp"), ckpt)
        escribir_json(d / "estado.json", leer_json(d / "estado.json", {}) | {"paso": paso, "actualizado": time.time()})

    n_tr = len(idx_tr)
    pasos_ep = math.ceil(n_tr / cfg["batch"])
    perm_ep, perm = -1, None
    t0, t_ckpt, suma, n_suma = time.time(), time.time(), torch.zeros((), device=device), 0
    model.train()
    while paso < objetivo:
        ep, j = divmod(paso, pasos_ep)
        if ep != perm_ep:   # el orden de los lotes depende sólo de (semilla, época): retomar repite el mismo orden
            perm, perm_ep = np.random.default_rng([cfg["seed"], ep]).permutation(n_tr), ep
        b = torch.tensor(idx_tr[perm[j * cfg["batch"]:(j + 1) * cfg["batch"]]], device=device)
        for g in opt.param_groups:
            g["lr"] = lr_en(paso, cfg)
        logits = model(Xt[b])
        ce = F.cross_entropy(logits.reshape(-1, V), Xt[b].reshape(-1), reduction="none").view(len(b), -1).mean(1)
        loss = (ce * wt[b]).sum() / wt[b].sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        paso += 1
        suma += loss.detach()
        n_suma += 1

        if paso % cfg["eval_cada"] == 0 or paso == objetivo:
            m, z, R = evaluar(model, u, val, et, cfg, device)
            fila = {"paso": paso, "paso_inicio": paso_inicio, "seg": time.time() - t0, "loss": (suma / n_suma).item(), "lr": lr_en(paso - 1, cfg)} | m
            with open(d / "metricas.jsonl", "a") as f:
                f.write(json.dumps(fila) + "\n")
            suma, n_suma = torch.zeros((), device=device), 0
            print(f"paso {paso:>7,}  loss {fila['loss']:.4f}" + (f"  S {m['S']:.4f}  S_sup {m['S_sup']:.4f}" if "S" in m else "")
                  + f"  rec {m.get('rec_anio', float('nan')):.4f}  ({fila['seg']:.0f}s)", flush=True)
            if paso == objetivo:
                np.savez_compressed(d / "codigos.npz", z=z.astype(np.float32), R=R, paso=paso)
                torch.save(model.state_dict(), d / "modelo.pt")
            guardar()
            t_ckpt = time.time()
        elif time.time() - t_ckpt > args.ckpt_seg:
            guardar()
            t_ckpt = time.time()
        if paso % 50 == 0 and parada.mirar() and paso < objetivo:
            guardar()
            escribir_json(d / "estado.json", leer_json(d / "estado.json") | {"estado": "interrumpida"})
            print(f"interrumpida en el paso {paso:,}", flush=True)
            sys.exit(SALIDA_INTERRUMPIDA)
    escribir_json(d / "estado.json", leer_json(d / "estado.json", {}) | {"estado": "terminada", "paso": paso, "fin": time.time()})


# ---------------------------------------------------------------------------
# Lanzador (reanudable, idempotente)
# ---------------------------------------------------------------------------
def vivo(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def en_curso(est: dict) -> bool:
    """¿La corrida sigue viva? En este nodo, por su pid. En otro nodo (cluster con /home compartido) el pid no se puede
    mirar: se la da por viva si actualizó estado.json hace menos de 3 intervalos de checkpoint."""
    if est.get("nodo", NODO) == NODO:
        return vivo(est.get("pid"))
    return time.time() - max(est.get("inicio", 0), est.get("actualizado", 0)) < 3 * est.get("ckpt_seg", 300)


def preparar(etapa: str, nombre: str, cfg: dict) -> Path:
    "Crea la carpeta de la corrida o sube su objetivo; el resto de la configuración no puede cambiar."
    d = MODELS / etapa / nombre
    d.mkdir(parents=True, exist_ok=True)
    viejo = leer_json(d / "config.json")
    if viejo is not None:
        distinto = {k for k in cfg if k != "objetivo" and viejo.get(k) != cfg[k]}
        assert not distinto, f"{nombre}: la configuración guardada difiere en {distinto}"
        cfg = viejo | {"objetivo": max(viejo["objetivo"], cfg["objetivo"])}
    escribir_json(d / "config.json", cfg)
    return d


def pendiente(d: Path) -> bool:
    est, cfg = leer_json(d / "estado.json", {}), leer_json(d / "config.json")
    return not (est.get("estado") == "terminada" and est.get("paso", 0) >= cfg["objetivo"])


def plazo(args) -> float:
    "Hora (epoch) a la que hay que parar, o inf."
    if args.max_horas:
        return time.time() + 3600 * args.max_horas
    if args.hasta:
        h, m = map(int, args.hasta.split(":"))
        t = dt.datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
        if t <= dt.datetime.now():
            t += dt.timedelta(days=1)
        return t.timestamp()
    return math.inf


def lanzar(dirs: list[Path], args) -> bool:
    "Corre las corridas pendientes con --workers en paralelo. True si quedaron todas terminadas."
    cola = []
    for d in dirs:
        if not pendiente(d):
            continue
        est = leer_json(d / "estado.json", {})
        if est.get("estado") == "en curso" and en_curso(est) and est.get("pid") != os.getpid():
            print(f"  {d.name}: ya está corriendo (pid {est['pid']} en {est.get('nodo', NODO)}), se saltea")
            continue
        cola.append(d)
    if not cola:
        return not any(pendiente(d) for d in dirs)
    devices = itertools.cycle(args.device.split(","))
    print(f"{len(cola)} corridas pendientes, {args.workers} a la vez", flush=True)
    activos, fin, cortar = {}, args.plazo, False
    try:
        while cola or activos:
            if not cortar and (STOP.exists() or time.time() >= fin):
                cortar = True
                print("parada pedida (" + ("STOP" if STOP.exists() else "horario") + "): no se lanzan corridas nuevas", flush=True)
                for p in activos.values():
                    p.send_signal(signal.SIGTERM)
            while not cortar and cola and len(activos) < args.workers:
                d = cola.pop(0)
                cmd = [sys.executable, str(Path(__file__).resolve()), "correr", "--dir", str(d), "--device", next(devices),
                       "--threads", str(args.threads), "--ckpt-seg", str(args.ckpt_seg)]
                activos[d] = subprocess.Popen(cmd, stdout=open(d / "log.txt", "a"), stderr=subprocess.STDOUT)
                print(f"  {time.strftime('%H:%M')} lanza {d.parent.name}/{d.name}", flush=True)
            for d, p in list(activos.items()):
                rc = p.poll()
                if rc is not None:
                    del activos[d]
                    txt = {0: "terminada", SALIDA_INTERRUMPIDA: "interrumpida"}.get(rc, f"ERROR (código {rc}; ver {P.rel(d / 'log.txt')})")
                    print(f"  {time.strftime('%H:%M')} {d.parent.name}/{d.name}: {txt}", flush=True)
            if cortar:
                cola.clear()
            time.sleep(2)
    except KeyboardInterrupt:   # Ctrl+C llega también a las corridas (mismo grupo de procesos): se espera a que guarden
        print("\nCtrl+C: esperando a que las corridas guarden su checkpoint...", flush=True)
        for p in activos.values():
            p.wait()
        sys.exit(SALIDA_INTERRUMPIDA)
    return not any(pendiente(d) for d in dirs)


def ultima_metrica(d: Path, paso: int | None = None) -> dict | None:
    f = d / "metricas.jsonl"
    if not f.exists():
        return None
    filas = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    if paso is not None:
        filas = [r for r in filas if r["paso"] == paso]
    return filas[-1] if filas else None


def preparar_lanzador(args):
    MODELS.mkdir(parents=True, exist_ok=True)
    if STOP.exists():
        STOP.unlink()
        print("se borró un STOP de una parada anterior")
    args.plazo = plazo(args)


# ---------------------------------------------------------------------------
# Etapas (§6)
# ---------------------------------------------------------------------------
def cfg_corrida(conf: dict, d: int, seed: int, s_max: int, objetivo: int, universo="entrenamiento") -> dict:
    return {k: conf[k] for k in ESPACIO} | {"d": d, "seed": seed, "s_max": s_max, "objetivo": objetivo,
                                           "universo": universo, "eval_cada": EVAL_CADA}


def cmd_piloto(args):
    preparar_lanzador(args)
    d = preparar("piloto", "base_d8_s0", cfg_corrida(BASE, D_BUSQUEDA, 0, PILOTO_PASOS, PILOTO_PASOS))
    print("piloto", "terminado" if lanzar([d], args) else "sin terminar (se retoma con el mismo comando)")


def plan_busqueda(s_max: int | None) -> dict:
    f = DATA / "plan_busqueda.json"
    plan = leer_json(f)
    if plan is None:
        assert s_max, "primera vez: pasar --s-max (sale del piloto: `tuneo_ae.py resumen`)"
        grilla = [dict(zip(ESPACIO, v)) for v in itertools.product(*ESPACIO.values())]
        grilla = [g for g in grilla if g != BASE]
        rng = np.random.default_rng(0)
        elegidas = [grilla[i] for i in rng.choice(len(grilla), N_BUSQUEDA - 1, replace=False)]
        plan = {"s_max": s_max, "configs": {f"c{i:02d}": c for i, c in enumerate([BASE] + elegidas)}}
        DATA.mkdir(parents=True, exist_ok=True)
        escribir_json(f, plan)
        print(f"plan de búsqueda guardado en {P.rel(f)} (S_max = {s_max:,})")
    elif s_max and s_max != plan["s_max"]:
        sys.exit(f"el plan ya existe con S_max = {plan['s_max']:,}; no se puede cambiar a {s_max:,}")
    return plan


def rondas(s_max: int):
    """(objetivo, cuántas corridas) de cada ronda. Una sola: las 24 hasta S_max (desviación 3 de tuneo.md: con rondas
    pasaban las configuraciones menos entrenadas, porque S baja con el entrenamiento)."""
    return [(s_max, N_BUSQUEDA)]


def ranking(plan: dict, ids: list[str], paso: int) -> list[str]:
    "Ids ordenados por S en ese paso (mayor primero; empates por id)."
    s = {}
    for c in ids:
        m = ultima_metrica(MODELS / "busqueda" / f"{c}_d{D_BUSQUEDA}_s0", paso)
        assert m is not None, f"{c}: falta la evaluación del paso {paso}"
        s[c] = m["S"]
    return sorted(ids, key=lambda c: (-s[c], c))


def cmd_busqueda(args):
    preparar_lanzador(args)
    plan = plan_busqueda(args.s_max)
    ids = list(plan["configs"])
    for r, (obj, n) in enumerate(rondas(plan["s_max"])):
        if r > 0:
            ids = ranking(plan, ids, rondas(plan["s_max"])[r - 1][0])[:n]
        print(f"ronda {r + 1}: {len(ids)} configuraciones hasta {obj:,} pasos", flush=True)
        dirs = [preparar("busqueda", f"{c}_d{D_BUSQUEDA}_s0", cfg_corrida(plan["configs"][c], D_BUSQUEDA, 0, plan["s_max"], obj))
                for c in ids]
        if not lanzar(dirs, args):
            print("búsqueda sin terminar (se retoma con el mismo comando)")
            return
    print("búsqueda terminada; mejores:", ranking(plan, ids, plan["s_max"]))


def finalistas(plan: dict) -> list[str]:
    s_max = plan["s_max"]
    ultimos = rondas(s_max)[-1]
    con_smax = [c for c in plan["configs"] if ultima_metrica(MODELS / "busqueda" / f"{c}_d{D_BUSQUEDA}_s0", s_max)]
    assert len(con_smax) == ultimos[1], "la búsqueda no terminó: correr `tuneo_ae.py busqueda`"
    return ranking(plan, con_smax, s_max)[:N_CONFIRMAR]


def cmd_confirmar(args):
    preparar_lanzador(args)
    plan = plan_busqueda(None)
    ids = finalistas(plan)
    escribir_json(DATA / "plan_confirmacion.json", {"configs": ids, "d": D_CONFIRMAR, "semillas": SEMILLAS})
    dirs = [preparar("confirmacion", f"{c}_d{dd}_s{s}", cfg_corrida(plan["configs"][c], dd, s, plan["s_max"], plan["s_max"]))
            for c in ids for dd in D_CONFIRMAR for s in SEMILLAS]
    print("confirmación", "terminada" if lanzar(dirs, args) else "sin terminar (se retoma con el mismo comando)")


def cmd_final(args):
    preparar_lanzador(args)
    plan = plan_busqueda(None)
    assert args.config in plan["configs"], f"{args.config} no está en el plan de búsqueda"
    dirs = [preparar("final", f"{args.config}_d{dd}_s{s}",
                     cfg_corrida(plan["configs"][args.config], dd, s, plan["s_max"], plan["s_max"], universo="todo"))
            for dd in D_FINAL for s in SEMILLAS]
    print("modelos finales", "terminados" if lanzar(dirs, args) else "sin terminar (se retoma con el mismo comando)")


# ---------------------------------------------------------------------------
# Estado y resumen
# ---------------------------------------------------------------------------
def cmd_estado(args):
    filas = []
    for f in sorted(MODELS.glob("*/*/config.json")):
        d = f.parent
        cfg, est, m = leer_json(f), leer_json(d / "estado.json", {}), ultima_metrica(d)
        e = est.get("estado", "pendiente")
        if e == "en curso":
            e = (f"en curso ({est['nodo']})" if est.get("nodo", NODO) != NODO else e) if en_curso(est) else "interrumpida (corte abrupto)"
        paso = est.get("paso", 0)
        # velocidad de la última sesión (desde que se lanzó o retomó)
        dt_ = est.get("actualizado", 0) - est.get("inicio", 0)
        vel = (paso - est.get("paso_inicio", 0)) / dt_ if dt_ > 0 and paso > est.get("paso_inicio", 0) else np.nan
        filas.append({"etapa": d.parent.name, "corrida": d.name, "estado": e, "paso": paso, "objetivo": cfg["objetivo"],
                      "S": m.get("S", np.nan) if m else np.nan, "pasos/s": vel,
                      "falta (h)": (cfg["objetivo"] - paso) / vel / 3600 if vel == vel and vel > 0 else np.nan})
    if not filas:
        print("no hay corridas todavía")
        return
    df = pd.DataFrame(filas)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n{(df.estado == 'terminada').sum()} terminadas de {len(df)}" + ("; hay un STOP pendiente" if STOP.exists() else ""))


def metricas_etapa(etapa: str) -> pd.DataFrame:
    filas = []
    for f in sorted((MODELS / etapa).glob("*/metricas.jsonl")):
        cfg = leer_json(f.parent / "config.json")
        for x in f.read_text().splitlines():
            if x.strip():
                filas.append({"corrida": f.parent.name, "config": f.parent.name.split("_d")[0], "d": cfg["d"],
                              "seed": cfg["seed"]} | json.loads(x))
    return pd.DataFrame(filas)


def eleccion(m: pd.DataFrame, s_max: int) -> pd.DataFrame:
    "§7: S y S_sup por configuración (media sobre d de la media entre semillas), rango entre semillas, parámetros."
    f = m[m.paso == s_max]
    por_semilla = f.groupby(["config", "seed"])[["S", "S_sup"]].mean()   # media sobre los d de cada semilla
    t = por_semilla.groupby("config").agg(S=("S", "mean"), S_min=("S", "min"), S_max_=("S", "max"),
                                          S_sup=("S_sup", "mean"), S_sup_min=("S_sup", "min"), S_sup_max=("S_sup", "max"))
    t["rango_S"] = t.S_max_ - t.S_min
    plan = plan_busqueda(None)
    t["parametros_d8"] = [n_parametros(plan["configs"][c] | {"d": 8}) for c in t.index]
    t["lote"] = [plan["configs"][c]["batch"] for c in t.index]
    mejor = t.S.idxmax()
    t["empata_con_mejor"] = t.S >= t.S[mejor] - t.rango_S[mejor]
    t["descartada_por_superficie"] = t.S_sup < t.S_sup[mejor] - (t.S_sup_max[mejor] - t.S_sup_min[mejor])
    cand = t[t.empata_con_mejor & ~t.descartada_por_superficie].sort_values(["parametros_d8", "lote"])
    t["elegida"] = t.index == (cand.index[0] if len(cand) else mejor)
    return t.sort_values("S", ascending=False)


def cmd_resumen(args):
    DATA.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    for etapa in ("piloto", "busqueda", "confirmacion", "final"):
        m = metricas_etapa(etapa)
        if m.empty:
            continue
        m.to_csv(DATA / f"metricas_{etapa}.csv", index=False)
        print(f"\n=== {etapa}: {m.corrida.nunique()} corridas -> {P.rel(DATA / f'metricas_{etapa}.csv')}")
        if etapa == "piloto":
            print(m[["paso", "loss", "S", "S_sup", "rec_anio", "rec_exacta"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            if m.paso.max() >= PILOTO_PASOS:
                # S_max por la reconstrucción (desviación 2 de tuneo.md): S baja con el entrenamiento
                tope = 0.99 * m.rec_exacta.max()
                c = m[(m.paso % 10_000 == 0) & (m.rec_exacta >= tope)].paso.min()
                u = m.iloc[-1]
                vel = (u.paso - u.paso_inicio) / u.seg
                print(f"velocidad {vel:.1f} pasos/s con un proceso; máximo de S {m.S.max():.4f} en el paso {m.paso[m.S.idxmax()]:,}")
                print(f"S_max = {int(min(max(c, 20_000), PILOTO_PASOS)):,} (primer múltiplo de 10.000 con secuencias exactas en "
                      f"validación >= 99 % del máximo, {tope:.4f})")
        elif etapa == "busqueda":
            plan = plan_busqueda(None)
            for obj, _ in rondas(plan["s_max"]):
                r = m[m.paso == obj].sort_values("S", ascending=False)
                if len(r):
                    print(f"-- {obj:,} pasos")
                    print(r[["config", "S", "S_sup", "rec_anio"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        elif etapa == "confirmacion":
            plan = plan_busqueda(None)
            f = m[m.paso == plan["s_max"]]
            print(f.groupby(["config", "d"]).S.agg(["mean", "min", "max", "count"]).unstack("d").round(4).to_string())
            if f.groupby("config").size().min() == len(D_CONFIRMAR) * len(SEMILLAS):
                t = eleccion(m, plan["s_max"])
                t.to_csv(DATA / "eleccion.csv")
                print(t.round(4).to_string())
                print("configuración elegida según §7:", t.index[t.elegida][0], "(revisar antes de `final --config`)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("particion").set_defaults(fn=cmd_particion)
    sub.add_parser("referencias").set_defaults(fn=cmd_referencias)
    sub.add_parser("estado").set_defaults(fn=cmd_estado)
    sub.add_parser("resumen").set_defaults(fn=cmd_resumen)
    sp = sub.add_parser("correr")
    sp.add_argument("--dir", required=True)
    sp.set_defaults(fn=cmd_correr)
    for nombre, fn in (("piloto", cmd_piloto), ("busqueda", cmd_busqueda), ("confirmar", cmd_confirmar), ("final", cmd_final)):
        sp = sub.add_parser(nombre)
        sp.set_defaults(fn=fn)
        sp.add_argument("--workers", type=int, default=1)
        sp.add_argument("--max-horas", type=float, default=None)
        sp.add_argument("--hasta", default=None, help="HH:MM (hoy, o mañana si ya pasó)")
        if nombre == "busqueda":
            sp.add_argument("--s-max", type=int, default=None, help="sólo la primera vez (sale del piloto)")
        if nombre == "final":
            sp.add_argument("--config", required=True, help="id de la configuración elegida (p. ej. c07)")
    for sp in sub.choices.values():
        if sp.get_default("fn") in (cmd_correr, cmd_piloto, cmd_busqueda, cmd_confirmar, cmd_final):
            sp.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0,cuda:1 (en rueda)")
            sp.add_argument("--threads", type=int, default=2)
            sp.add_argument("--ckpt-seg", type=int, default=300, help="segundos entre checkpoints")
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
