#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build/load a stratified fixed-500 subset from test split, then evaluate
subset-vs-full Pearson correlation across candidate model result files.

Expected per-job outputs from sweet_spot_run_lora_screener_GA.py (ESM mode):
- *_per_item_s2.csv: columns [id, long_range_pl, length]
- *.json: long_range_pl.full_test_s2
"""

import argparse
import csv
import json
from pathlib import Path
import numpy as np


def _length_bin(L):
    if L < 256:
        return "<256"
    if L <= 512:
        return "256-512"
    return ">512"


def stratified_sample(ids_and_len, n, seed=42):
    if n > len(ids_and_len):
        raise ValueError(f"n={n} > data={len(ids_and_len)}")
    by_bin = {"<256": [], "256-512": [], ">512": []}
    for i, (_, L) in enumerate(ids_and_len):
        by_bin[_length_bin(int(L))].append(i)

    rng = np.random.RandomState(seed)
    target = {}
    for k in by_bin:
        target[k] = int(round(len(by_bin[k]) / max(1, len(ids_and_len)) * n))

    while sum(target.values()) < n:
        k = max(by_bin.keys(), key=lambda x: len(by_bin[x]) - target[x])
        target[k] += 1
    while sum(target.values()) > n:
        k = max(target.keys(), key=lambda x: target[x])
        if target[k] > 0:
            target[k] -= 1

    picked = []
    for k, pool in by_bin.items():
        take = min(target[k], len(pool))
        if take > 0:
            picked.extend(rng.choice(pool, size=take, replace=False).tolist())

    if len(picked) < n:
        remaining = sorted(set(range(len(ids_and_len))) - set(picked))
        picked.extend(rng.choice(remaining, size=n - len(picked), replace=False).tolist())
    picked = sorted(picked)
    return picked


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def load_per_item(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "id": str(r["id"]),
                "long_range_pl": float(r["long_range_pl"]),
                "length": int(float(r["length"])),
            })
    return rows


def subset_mean(rows, subset_ids):
    wanted = set(subset_ids)
    vals = [r["long_range_pl"] for r in rows if r["id"] in wanted and not np.isnan(r["long_range_pl"])]
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def collect_candidate_files(results_dir):
    results_dir = Path(results_dir)
    json_files = sorted([p for p in results_dir.glob("*.json") if not p.name.endswith("split_manifest.json")])
    items = []
    for jp in json_files:
        if jp.name.endswith("fixed_subset_500.json"):
            continue
        try:
            rec = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        lp = rec.get("long_range_pl", {})
        full = lp.get("full_test_s2")
        csv_path = lp.get("full_test_s2_per_item_csv")
        if full is None:
            continue
        if csv_path is None:
            guess = jp.with_name(jp.stem + "_per_item_s2.csv")
            csv_path = str(guess)
        items.append({"json": jp, "full": float(full), "csv": Path(csv_path)})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument("--fixed_subset_manifest", type=str, default=None)
    ap.add_argument("--subset_size", type=int, default=500)
    ap.add_argument("--random_trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report_out", type=str, default=None)
    args = ap.parse_args()

    cands = collect_candidate_files(args.results_dir)
    if len(cands) == 0:
        raise RuntimeError("No candidate result files found.")

    base_rows = load_per_item(cands[0]["csv"])
    ids_and_len = [(r["id"], r["length"]) for r in base_rows]

    if args.fixed_subset_manifest:
        fixed_manifest = Path(args.fixed_subset_manifest)
    else:
        fixed_manifest = Path(args.results_dir) / "fixed_subset_500.json"

    if fixed_manifest.exists():
        payload = json.loads(fixed_manifest.read_text(encoding="utf-8"))
        fixed_ids = payload["ids"]
    else:
        idx = stratified_sample(ids_and_len, args.subset_size, seed=args.seed)
        fixed_ids = [ids_and_len[i][0] for i in idx]
        fixed_manifest.write_text(json.dumps({"seed": args.seed, "ids": fixed_ids, "size": len(fixed_ids)}, indent=2), encoding="utf-8")

    full_vec = []
    fixed_vec = []
    random_vecs = [[] for _ in range(args.random_trials)]

    trial_subsets = []
    for t in range(args.random_trials):
        idx = stratified_sample(ids_and_len, args.subset_size, seed=args.seed + 1000 + t)
        trial_subsets.append([ids_and_len[i][0] for i in idx])

    for c in cands:
        rows = load_per_item(c["csv"])
        full_vec.append(float(c["full"]))
        fixed_vec.append(subset_mean(rows, fixed_ids))
        for t in range(args.random_trials):
            random_vecs[t].append(subset_mean(rows, trial_subsets[t]))

    fixed_r = pearson(full_vec, fixed_vec)
    random_rs = [pearson(full_vec, rv) for rv in random_vecs]

    report = {
        "results_dir": str(Path(args.results_dir).resolve()),
        "num_candidates": len(cands),
        "subset_size": args.subset_size,
        "random_trials": args.random_trials,
        "threshold": 0.9,
        "fixed_subset_manifest": str(fixed_manifest),
        "fixed_subset_pearson_r": fixed_r,
        "random_subset_pearson_r": random_rs,
        "random_subset_mean_r": float(np.nanmean(random_rs)) if len(random_rs) else float("nan"),
        "is_effective": bool((not np.isnan(fixed_r)) and fixed_r > 0.9),
    }

    out = Path(args.report_out) if args.report_out else Path(args.results_dir) / "subset_corr_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
