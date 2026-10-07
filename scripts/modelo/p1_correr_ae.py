"""Lanza en paralelo los autoencoders de P1 (d × semilla) llamando a `p1_compresion.py train`.

Default: d ∈ DIMS de p1_compresion (1, 4, 7, ..., 31) × semillas 0, 1, 2 = 33 modelos.
Es reanudable: salta los modelos cuyo models/autoencoder_v3/p1/ae_d<d>_s<s>.npz ya existe, así que
si se corta se vuelve a correr el mismo comando. Un log por modelo en --log-dir.

    python scripts/modelo/p1_correr_ae.py --workers 3                   # GPU (auto), 3 modelos a la vez
    python scripts/modelo/p1_correr_ae.py --device cpu --workers 4 --threads 2
    python scripts/modelo/p1_correr_ae.py --dims 13 16 19 --seeds 1 2   # sólo un subconjunto
    python scripts/modelo/p1_correr_ae.py --dry-run                     # lista lo que correría

Con varias GPUs, --device cuda:0,cuda:1 reparte los modelos en rueda entre ellas.
Los demás argumentos (--epochs, --batch, --lr, ...) se pasan tal cual a `train`: ponerlos después de `--`.
"""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p1_compresion as p1  # noqa: E402


def run_one(job):
    d, seed, device, threads, extra, log_dir = job
    log = log_dir / f"ae_d{d}_s{seed}.log"
    cmd = [sys.executable, str(HERE / "p1_compresion.py"), "train", "--d", str(d), "--seed", str(seed),
           "--device", device, "--threads", str(threads), *extra]
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    return d, seed, rc, time.time() - t0, log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dims", type=int, nargs="+", default=p1.DIMS)
    ap.add_argument("--seeds", type=int, nargs="+", default=p1.SEEDS)
    ap.add_argument("--workers", type=int, default=3, help="modelos simultáneos (GPU: 2-4; CPU: n.º de núcleos / --threads)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0,cuda:1 (reparte en rueda)")
    ap.add_argument("--threads", type=int, default=1, help="hilos de CPU por modelo")
    ap.add_argument("--log-dir", type=Path, default=p1.OUT_MODELS / "logs")
    ap.add_argument("--dry-run", action="store_true")
    args, extra = ap.parse_known_args()
    extra = [e for e in extra if e != "--"]

    todo = [(d, s) for s in args.seeds for d in args.dims
            if not (p1.OUT_MODELS / f"ae_d{d}_s{s}.npz").exists()]
    total = len(args.dims) * len(args.seeds)
    print(f"{total} modelos pedidos, {total - len(todo)} ya hechos, {len(todo)} por correr "
          f"({args.workers} en paralelo, device={args.device})", flush=True)
    if args.dry_run or not todo:
        for d, s in todo:
            print(f"  d={d} semilla={s}")
        return
    args.log_dir.mkdir(parents=True, exist_ok=True)
    devices = cycle(args.device.split(","))
    jobs = [(d, s, next(devices), args.threads, extra, args.log_dir) for d, s in todo]

    t0, ok, fail = time.time(), 0, []
    with ThreadPoolExecutor(args.workers) as ex:
        for d, s, rc, dt, log in ex.map(run_one, jobs):
            tail = log.read_text().strip().splitlines()[-1:] or [""]
            if rc == 0:
                ok += 1
            else:
                fail.append((d, s))
            print(f"[{ok + len(fail)}/{len(todo)}] d={d} s={s} {'ok' if rc == 0 else f'ERROR rc={rc}'} "
                  f"{dt / 60:.1f} min  {tail[0]}  (transcurrido {(time.time() - t0) / 60:.0f} min)", flush=True)
    print(f"\nterminado: {ok} ok, {len(fail)} con error {fail}" if fail else f"\nterminado: {ok} ok", flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
