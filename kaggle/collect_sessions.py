#!/usr/bin/env python3
"""Merge the per-session summaries of the PACS grid into one table.

Usage (after both Kaggle sessions finished):

    for k in ggn-pacs-grid ggn-pacs-grid-env23; do
        kaggle kernels output ducle19/$k -p results/$k -q
    done
    python kaggle/collect_sessions.py results/*

Reads every grid_summary.json below the given directories and prints the
test-domain accuracy of each algorithm per test env, plus the cost of each run.
"""
import glob
import json
import os
import statistics
import sys

ENVS = ["A art_painting", "C cartoon", "P photo", "S sketch"]

runs = {}
for root in sys.argv[1:] or ["."]:
    for path in glob.glob(os.path.join(root, "**", "grid_summary.json"),
                          recursive=True) + glob.glob(
                              os.path.join(root, "grid_summary.json")):
        summary = json.load(open(path))
        print(f"# {path}: steps={summary.get('steps')} "
              f"envs={summary.get('env_subset')} "
              f"ckpt={summary.get('checkpoint_freq')}")
        for name, run in summary.get("runs", {}).items():
            runs[name] = run

if not runs:
    raise SystemExit("no grid_summary.json found")

hdr = (f"{'test env':16s} {'algorithm':10s} {'status':14s} {'test_in':>8s}"
       f" {'test_out':>9s} {'train_out':>10s} {'wall_h':>7s} {'s/step':>7s}"
       f" {'mem_gb':>7s} {'ggn':>9s}")
print("\n" + hdr + "\n" + "-" * len(hdr))


def f(v, w, p=4):
    return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}s}"


for env in range(4):
    for algo in ("GGNMatch", "ERM"):
        run = runs.get(f"{algo}_env{env}")
        if run is None:
            continue
        print(f"{ENVS[env]:16s} {algo:10s} {run.get('status', '?'):14s}"
              f" {f(run.get('test_in_acc'), 8)} {f(run.get('test_out_acc'), 9)}"
              f" {f(run.get('train_out_acc_mean'), 10)}"
              f" {f(run.get('wall_h'), 7, 2)} {f(run.get('step_time_s'), 7, 3)}"
              f" {f(run.get('mem_gb'), 7, 2)} {f(run.get('ggn'), 9, 3)}")

print()
for algo in ("GGNMatch", "ERM"):
    accs = [(e, runs[f"{algo}_env{e}"]["test_in_acc"]) for e in range(4)
            if f"{algo}_env{e}" in runs
            and runs[f"{algo}_env{e}"].get("test_in_acc") is not None]
    if accs:
        mean = statistics.mean(a for _, a in accs)
        detail = "  ".join(f"{ENVS[e].split()[0]}={a:.4f}" for e, a in accs)
        print(f"{algo:10s} mean over {len(accs)} env(s) = {mean:.4f}   {detail}")
