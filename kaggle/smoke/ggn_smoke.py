#!/usr/bin/env python3
"""GGNMatch smoke test on PACS -- one Kaggle GPU session, ~20 minutes.

Validates, in order: the Kaggle environment, the repo clone, dependencies, the
PACS data wiring, the ggn unit tests on Kaggle's torch build, and three short
training runs (ERM baseline, GGNMatch with 11 and with 4 matched layers).

It reports per-step time, peak VRAM and the penalty values, then extrapolates
the cost of the full 5001-step PACS grid so the real session can be sized.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

REPO = "https://github.com/kuduck192/domain_gen.git"
CLONE = "/tmp/domain_gen"
SRC = CLONE              # set after the clone: the repo root holds domainbed/
WORK = "/kaggle/working"
OUT = os.path.join(WORK, "smoke")
DATA = "/tmp/data"
WORKERS = "2"            # Kaggle has ~4 vCPUs and 11 persistent DataLoaders
STEPS = 101
CKPT_FREQ = 100          # evals at step 0 and 100
ANNEAL = 0               # penalty active from the first step
RUN_TIMEOUT = 40 * 60

DOMAINS = ("art_painting", "cartoon", "photo", "sketch")

t_session = time.time()


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
    print(f"\n{'=' * 78}\n[{time.time() - t_session:7.1f}s] {msg}\n{'=' * 78}",
          flush=True)


def sh(cmd, cwd=None, env=None, timeout=None, check=True):
    """Run a command, streaming its output into the Kaggle log."""
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
sh("free -g; nproc; df -h /tmp /kaggle/working", check=False)
sh(f"{sys.executable} -c \"import torch, torchvision, timm;"
   f" print('torch', torch.__version__, 'tv', torchvision.__version__,"
   f" 'timm', timm.__version__);"
   f" print('cuda', torch.version.cuda, torch.cuda.get_device_name(0))\"")

# ---------------------------------------------------------------------- clone
say("Clone repo")
shutil.rmtree(CLONE, ignore_errors=True)
sh(f"git clone --depth 1 -b ggn-position-gram {REPO} {CLONE}")
sh("git log --oneline -1", cwd=CLONE)
# The repo root may itself be the DomainBed tree, or hold it in a subdirectory.
for cand in (CLONE, os.path.join(CLONE, "DomainBed")):
    if os.path.isdir(os.path.join(cand, "domainbed")) and \
            os.path.isdir(os.path.join(cand, "ggn")):
        SRC = cand
        break
else:
    sh(f"find {CLONE} -maxdepth 2 -type d | head -30", check=False)
    raise SystemExit("could not locate the domainbed/ + ggn/ tree in the clone")
print(f"source tree: {SRC}", flush=True)

# --------------------------------------------------------------- dependencies
say("Dependencies")
# domainbed/datasets.py imports wilds at module level (only for the WILDS
# datasets). Install it if missing; fall back to making the import optional.
if subprocess.run([sys.executable, "-c", "import wilds"],
                  capture_output=True).returncode != 0:
    sh(f"{sys.executable} -m pip install -q wilds", check=False)
if subprocess.run([sys.executable, "-c", "import wilds"],
                  capture_output=True).returncode != 0:
    print("wilds unavailable -> making its import optional", flush=True)
    path = os.path.join(SRC, "domainbed", "datasets.py")
    text = open(path).read()
    old = ("from wilds.datasets.camelyon17_dataset import Camelyon17Dataset\n"
           "from wilds.datasets.fmow_dataset import FMoWDataset\n")
    new = ("try:\n"
           "    from wilds.datasets.camelyon17_dataset import Camelyon17Dataset\n"
           "    from wilds.datasets.fmow_dataset import FMoWDataset\n"
           "except ImportError:  # WILDS datasets unused in this session\n"
           "    Camelyon17Dataset = FMoWDataset = None\n")
    assert old in text, "wilds import block not found"
    open(path, "w").write(text.replace(old, new))

# 11 persistent DataLoaders x N_WORKERS processes; 8 is too many on Kaggle.
path = os.path.join(SRC, "domainbed", "datasets.py")
text = open(path).read()
assert "N_WORKERS = 8" in text, "N_WORKERS default changed"
open(path, "w").write(text.replace(
    "N_WORKERS = 8",
    'N_WORKERS = int(os.environ.get("DOMAINBED_WORKERS", 8))', 1))
print(f"patched N_WORKERS -> env DOMAINBED_WORKERS={WORKERS}", flush=True)

# ------------------------------------------------------------------- the data
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
print(f"{link} -> {pacs}", flush=True)

total = 0
for domain in sorted(os.listdir(link)):
    classes = sorted(os.listdir(os.path.join(link, domain)))
    counts = {c: len(os.listdir(os.path.join(link, domain, c))) for c in classes}
    total += sum(counts.values())
    print(f"  {domain:14s} {sum(counts.values()):5d}  {counts}", flush=True)
print(f"  total {total} images (standard PACS = 9991)", flush=True)
if total != 9991:
    print("WARNING: image count differs from the standard PACS split", flush=True)

# ------------------------------------------------------------------ unit tests
say("ggn unit tests on the Kaggle torch build")
env = dict(os.environ, PYTHONPATH=SRC, DOMAINBED_WORKERS=WORKERS,
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
tests = sh(f"{sys.executable} -m unittest ggn.tests.test_ggn -v",
           cwd=SRC, env=env, timeout=1800, check=False)
tests_ok = tests.returncode == 0

# ----------------------------------------------------------------- the runs
# ERM (0.97 s/step, 8.1 GB) and GGNMatch with 4 layers (1.18 s/step, 11.2 GB)
# were measured in the previous session; this pass re-checks the 11-layer
# config, which used to run out of memory before layer_gram gained its second
# route.
RUNS = [
    ("ggn_layers11", "GGNMatch", {"ggn_anneal_iters": ANNEAL}),
]

summary = {
    "tests_ok": tests_ok,
    "steps": STEPS,
    "checkpoint_freq": CKPT_FREQ,
    "ggn_anneal_iters": ANNEAL,
    "workers": int(WORKERS),
    "runs": {},
}
os.makedirs(OUT, exist_ok=True)

for name, algorithm, hparams in RUNS:
    say(f"Train: {name} ({algorithm}, hparams={hparams})")
    outdir = os.path.join(OUT, name)
    cmd = (f"{sys.executable} -m domainbed.scripts.train"
           f" --data_dir {DATA} --dataset PACS --algorithm {algorithm}"
           f" --test_envs 0 --steps {STEPS} --checkpoint_freq {CKPT_FREQ}"
           f" --skip_model_save --output_dir {outdir}"
           f" --hparams '{json.dumps(hparams)}'")
    t0 = time.time()
    try:
        p = sh(cmd, cwd=SRC, env=env, timeout=RUN_TIMEOUT, check=False)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    wall = time.time() - t0

    rows = []
    jsonl = os.path.join(outdir, "results.jsonl")
    if os.path.exists(jsonl):
        rows = [json.loads(l) for l in open(jsonl) if l.strip()]
    summary["runs"][name] = {
        "algorithm": algorithm, "hparams": hparams, "returncode": rc,
        "wall_s": round(wall, 1), "ok": rc == 0 and len(rows) >= 2,
        "checkpoints": rows,
    }
    print(f"-> returncode={rc}, wall={wall:.1f}s, checkpoints={len(rows)}",
          flush=True)
    if rc != 0:
        err = os.path.join(outdir, "err.txt")
        if os.path.exists(err):
            print("--- tail of err.txt ---", flush=True)
            print("".join(open(err).readlines()[-40:]), flush=True)

# ------------------------------------------------------------------- summary
say("Summary")


def window(rows, step):
    """The checkpoint row whose averaging window ends at `step`."""
    for r in rows:
        if r["step"] == step:
            return r
    return None


hdr = (f"{'run':14s} {'ok':>4s} {'wall_s':>8s} {'first_s/it':>12s}"
       f" {'active_s/it':>12s} {'mem_gb':>7s} {'test_acc':>9s}"
       f" {'ggn':>10s} {'ggn_raw':>10s}")
print(hdr, flush=True)
print("-" * len(hdr), flush=True)
for name, info in summary["runs"].items():
    rows = info["checkpoints"]
    warm = window(rows, CKPT_FREQ)            # steps 1..100 (penalty off)
    act = window(rows, STEPS - 1)             # steps 101..200 (penalty on)
    mem = max([r.get("mem_gb", 0) for r in rows], default=0)
    acc = (act or {}).get("env0_in_acc")
    info["warmup_step_s"] = (warm or {}).get("step_time")
    info["active_step_s"] = (act or {}).get("step_time")
    info["mem_gb"] = mem
    info["test_acc"] = acc
    info["ggn"] = (act or {}).get("ggn")
    info["ggn_raw"] = (act or {}).get("ggn_raw")

    def f(v, w, p=4):
        return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}s}"

    print(f"{name:14s} {str(info['ok']):>4s} {info['wall_s']:8.1f}"
          f" {f(info['warmup_step_s'], 12)} {f(info['active_step_s'], 12)}"
          f" {f(mem, 7, 2)} {f(acc, 9, 4)}"
          f" {f(info['ggn'], 10, 3)} {f(info['ggn_raw'], 10, 3)}", flush=True)

# Cost model for the real grid: 5001 steps, evals every 300 steps. The eval
# cost is CPU-bound and the same for every algorithm, so take it from the
# first successful run.
ref = next((r for r in summary["runs"].values() if r.get("ok")), None)
base_eval = None
if ref and ref.get("active_step_s"):
    train_s = sum(r["step_time"] * CKPT_FREQ for r in ref["checkpoints"][1:])
    base_eval = max((ref["wall_s"] - train_s - 60.0) / len(ref["checkpoints"]), 1.0)
summary["eval_round_s_est"] = base_eval and round(base_eval, 1)

print(f"\nestimated eval round (8 splits, 9991 images): "
      f"{base_eval and round(base_eval)}s", flush=True)
print("\nextrapolation to the full PACS protocol "
      "(5001 steps, checkpoint_freq=300, one test env):", flush=True)
for name, info in summary["runs"].items():
    s = info.get("active_step_s")
    if not s or not base_eval:
        continue
    hours = (5001 * s + 17 * base_eval + 90) / 3600.0
    info["full_run_hours_est"] = round(hours, 2)
    print(f"  {name:14s} {hours:5.2f} h / env   ->  4 envs: {4 * hours:5.2f} h",
          flush=True)
if all(k in summary["runs"] and summary["runs"][k].get("full_run_hours_est")
       for k in ("erm_freezebn", "ggn_layers11")):
    grid = 4 * (summary["runs"]["erm_freezebn"]["full_run_hours_est"]
                + summary["runs"]["ggn_layers11"]["full_run_hours_est"])
    summary["grid_4env_ggn11_plus_erm_hours_est"] = round(grid, 2)
    print(f"  full grid (4 envs x [GGNMatch(11 layers) + ERM]): {grid:.2f} h",
          flush=True)
    ggn_s = summary["runs"]["ggn_layers11"]["active_step_s"]
    print(f"  GGNMatch overhead vs ERM per step: "
          f"{ggn_s / summary['runs']['erm_freezebn']['active_step_s']:.2f}x",
          flush=True)

with open(os.path.join(WORK, "smoke_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2),
      flush=True)

all_ok = tests_ok and all(i["ok"] for i in summary["runs"].values())
say(f"SMOKE {'PASSED' if all_ok else 'FAILED'} "
    f"(total {(time.time() - t_session) / 60:.1f} min)")
if not all_ok:
    raise SystemExit(1)
