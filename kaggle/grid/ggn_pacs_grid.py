#!/usr/bin/env python3
"""GGNMatch vs ERM on PACS -- leave-one-domain-out, one Kaggle GPU session.

Runs, in this order, (GGNMatch, ERM+freeze_bn) for test env 0, 1, 2, 3, so that
whatever finishes inside the session is always a complete per-domain pair. Each
run writes incrementally to /kaggle/working/grid/<name>/results.jsonl; a run is
skipped (and logged as such) when the remaining session budget cannot hold it.
Finishes with domainbed.scripts.collect_results over whatever completed.
"""
import glob
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

REPO = "https://github.com/kuduck192/domain_gen.git"
CLONE = "/tmp/domain_gen"
SRC = CLONE
WORK = "/kaggle/working"
OUT = os.path.join(WORK, "grid")
DATA = "/tmp/data"
WORKERS = "2"

STEPS = int(os.environ.get("GGN_STEPS", 5001))
CKPT_FREQ = int(os.environ.get("GGN_CKPT", 300))
BUDGET_S = float(os.environ.get("GGN_BUDGET_H", 11.0)) * 3600
# Prior per-run wall-clock estimates (seconds), replaced by measurements as
# soon as one run of that algorithm has finished. From the smoke session on a
# T4: 2.373 s/step for GGNMatch (11 layers, 11.1 GB) and 0.965 s/step for ERM,
# plus ~40 s per eval round over the eight PACS splits.
PRIOR_S = {"GGNMatch": float(os.environ.get("GGN_PRIOR_GGN", 5001 * 2.373 + 17 * 40 + 60)),
           "ERM": float(os.environ.get("GGN_PRIOR_ERM", 5001 * 0.965 + 17 * 40 + 60))}

DOMAINS = ("art_painting", "cartoon", "photo", "sketch")

t_session = time.time()


def left():
    return BUDGET_S - (time.time() - t_session)


def find_pacs():
    """Locate the PACS root under /kaggle/input, whatever the mount layout."""
    for depth in range(1, 6):
        pattern = "/kaggle/input/" + "/".join(["*"] * depth)
        for p in sorted(glob.glob(pattern)):
            if os.path.isdir(p) and all(
                    os.path.isdir(os.path.join(p, d)) for d in DOMAINS):
                return p
    return None


def say(msg):
    print(f"\n{'=' * 78}\n[{(time.time() - t_session) / 60:7.1f} min, "
          f"{left() / 3600:4.1f} h left] {msg}\n{'=' * 78}", flush=True)


def sh(cmd, cwd=None, env=None, timeout=None, check=True):
    print(f"$ {cmd}", flush=True)
    sys.stdout.flush()
    t = time.time()
    p = subprocess.run(cmd, shell=True, cwd=cwd, env=env, timeout=timeout,
                       stderr=subprocess.STDOUT)
    print(f"[exit {p.returncode} in {time.time() - t:.1f}s]", flush=True)
    if check and p.returncode != 0:
        raise SystemExit(f"command failed: {cmd}")
    return p


# ---------------------------------------------------------------- environment
say("Environment")
sh("nvidia-smi", check=False)
sh("nproc; free -g", check=False)
sh(f"{sys.executable} -c \"import torch; print('torch', torch.__version__,"
   f" torch.cuda.get_device_name(0))\"")

say("Clone repo")
shutil.rmtree(CLONE, ignore_errors=True)
sh(f"git clone --depth 1 -b ggn-position-gram {REPO} {CLONE}")
sh("git log --oneline -1", cwd=CLONE)
for cand in (CLONE, os.path.join(CLONE, "DomainBed")):
    if os.path.isdir(os.path.join(cand, "domainbed")) and \
            os.path.isdir(os.path.join(cand, "ggn")):
        SRC = cand
        break
else:
    raise SystemExit("could not locate the domainbed/ + ggn/ tree in the clone")
print(f"source tree: {SRC}", flush=True)

say("Dependencies")
if subprocess.run([sys.executable, "-c", "import wilds"],
                  capture_output=True).returncode != 0:
    sh(f"{sys.executable} -m pip install -q wilds", check=False)
path = os.path.join(SRC, "domainbed", "datasets.py")
text = open(path).read()
if subprocess.run([sys.executable, "-c", "import wilds"],
                  capture_output=True).returncode != 0:
    old = ("from wilds.datasets.camelyon17_dataset import Camelyon17Dataset\n"
           "from wilds.datasets.fmow_dataset import FMoWDataset\n")
    new = ("try:\n"
           "    from wilds.datasets.camelyon17_dataset import Camelyon17Dataset\n"
           "    from wilds.datasets.fmow_dataset import FMoWDataset\n"
           "except ImportError:\n"
           "    Camelyon17Dataset = FMoWDataset = None\n")
    assert old in text, "wilds import block not found"
    text = text.replace(old, new)
assert "N_WORKERS = 8" in text, "N_WORKERS default changed"
text = text.replace("N_WORKERS = 8",
                    'N_WORKERS = int(os.environ.get("DOMAINBED_WORKERS", 8))', 1)
open(path, "w").write(text)

say("PACS data")
pacs = find_pacs()
if pacs is None:
    sh("find /kaggle/input -maxdepth 4 -type d | head -40", check=False)
    raise SystemExit("PACS domain directories not found under /kaggle/input")
os.makedirs(DATA, exist_ok=True)
link = os.path.join(DATA, "PACS")
if os.path.islink(link) or os.path.exists(link):
    os.remove(link)
os.symlink(pacs, link)
total = len(glob.glob(os.path.join(link, "*", "*", "*")))
print(f"{link} -> {pacs} ({total} images)", flush=True)

# ------------------------------------------------------------------- the grid
ENVS = ["A(art_painting)", "C(cartoon)", "P(photo)", "S(sketch)"]
# One Kaggle session cannot hold all four test envs at 5001 steps, so the grid
# is split over two sessions. The second session is pushed as its own kernel
# with the default below rewritten to "2,3".
ENV_SUBSET = [int(e) for e in os.environ.get("GGN_ENVS", "0,1").split(",")]
RUNS = []
for e in ENV_SUBSET:
    RUNS.append((f"GGNMatch_env{e}", "GGNMatch", e, {}))
    RUNS.append((f"ERM_env{e}", "ERM", e, {"freeze_bn": True}))

env = dict(os.environ, PYTHONPATH=SRC, DOMAINBED_WORKERS=WORKERS,
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
os.makedirs(OUT, exist_ok=True)
summary = {"steps": STEPS, "checkpoint_freq": CKPT_FREQ,
           "env_subset": ENV_SUBSET,
           "budget_h": BUDGET_S / 3600, "runs": {}}
durations = {"GGNMatch": [], "ERM": []}


def save_summary():
    with open(os.path.join(WORK, "grid_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


for name, algorithm, test_env, hparams in RUNS:
    est = (statistics.median(durations[algorithm]) if durations[algorithm]
           else PRIOR_S[algorithm])
    if left() < est * 1.08 + 300:
        print(f"SKIP {name}: needs ~{est / 3600:.2f} h, "
              f"only {left() / 3600:.2f} h left", flush=True)
        summary["runs"][name] = {"status": "skipped_no_time",
                                 "estimate_h": round(est / 3600, 2)}
        save_summary()
        continue

    say(f"{name}  (test env {test_env} = {ENVS[test_env]}, "
        f"estimate {est / 3600:.2f} h)")
    outdir = os.path.join(OUT, name)
    cmd = (f"{sys.executable} -m domainbed.scripts.train"
           f" --data_dir {DATA} --dataset PACS --algorithm {algorithm}"
           f" --test_envs {test_env} --steps {STEPS}"
           f" --checkpoint_freq {CKPT_FREQ} --skip_model_save"
           f" --output_dir {outdir}")
    if hparams:
        cmd += f" --hparams '{json.dumps(hparams)}'"
    t0 = time.time()
    try:
        rc = sh(cmd, cwd=SRC, env=env, timeout=max(left() - 240, 60),
                check=False).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    wall = time.time() - t0
    if rc == 0:
        durations[algorithm].append(wall)

    rows = []
    jsonl = os.path.join(outdir, "results.jsonl")
    if os.path.exists(jsonl):
        rows = [json.loads(l) for l in open(jsonl) if l.strip()]
    last = rows[-1] if rows else {}
    summary["runs"][name] = {
        "status": "done" if rc == 0 else f"failed({rc})",
        "algorithm": algorithm, "test_env": test_env, "hparams": hparams,
        "wall_h": round(wall / 3600, 3), "checkpoints": len(rows),
        "last_step": last.get("step"),
        "step_time_s": last.get("step_time"),
        "mem_gb": max([r.get("mem_gb", 0) for r in rows], default=0),
        "test_in_acc": last.get(f"env{test_env}_in_acc"),
        "test_out_acc": last.get(f"env{test_env}_out_acc"),
        "train_out_acc_mean": (
            statistics.mean([last[f"env{i}_out_acc"] for i in range(4)
                             if i != test_env and f"env{i}_out_acc" in last])
            if last else None),
        "ggn": last.get("ggn"),
        "nll": last.get("nll"),
    }
    print(json.dumps(summary["runs"][name], indent=2), flush=True)
    save_summary()
    if rc != 0:
        err = os.path.join(outdir, "err.txt")
        if os.path.exists(err):
            print("--- tail of err.txt ---\n"
                  + "".join(open(err).readlines()[-40:]), flush=True)

# ------------------------------------------------------------------- results
say("Per-run summary")
hdr = (f"{'run':16s} {'status':14s} {'wall_h':>7s} {'step_s':>7s} {'mem_gb':>7s}"
       f" {'test_in':>8s} {'test_out':>9s} {'train_out':>10s}")
print(hdr + "\n" + "-" * len(hdr), flush=True)
for name, r in summary["runs"].items():
    def f(v, w, p=4):
        return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}s}"
    print(f"{name:16s} {r['status']:14s} {f(r.get('wall_h'), 7, 2)}"
          f" {f(r.get('step_time_s'), 7, 3)} {f(r.get('mem_gb'), 7, 2)}"
          f" {f(r.get('test_in_acc'), 8)} {f(r.get('test_out_acc'), 9)}"
          f" {f(r.get('train_out_acc_mean'), 10)}", flush=True)

done = [r for r in summary["runs"].values() if r.get("status") == "done"]
for algo in ("GGNMatch", "ERM"):
    accs = [r["test_in_acc"] for r in done
            if r.get("algorithm") == algo and r.get("test_in_acc") is not None]
    if accs:
        summary[f"{algo}_mean_test_acc"] = round(statistics.mean(accs), 4)
        print(f"\n{algo}: mean test-domain accuracy over {len(accs)} env(s) = "
              f"{statistics.mean(accs):.4f}  {[round(a, 4) for a in accs]}",
              flush=True)
save_summary()

say("collect_results (DomainBed model selection)")
sh(f"{sys.executable} -m domainbed.scripts.collect_results --input_dir {OUT}",
   cwd=SRC, env=env, check=False)

say(f"GRID FINISHED ({(time.time() - t_session) / 3600:.2f} h)")
