"""Milestone B: end-to-end chronological replay of all six policies.

    python -m experiments.run_replay --config configs/default.toml
    python -m experiments.run_replay --synthetic 3000 --max-dates 200   # labelled synthetic smoke run
    python -m experiments.run_replay --set decision.lam=5 --set decision.cost_A=1e-4 --tag lam5

Writes (prefix = results/<tag>):
  results/logs/<tag>_log.csv.gz        one row per (date, policy)
  results/tables/<tag>_summary.csv     per-policy summary (bp/day)
  results/tables/<tag>_paired.csv      paired differences vs free-only with block-bootstrap CIs
  results/tables/<tag>_manifest.json   config, hashes, seeds, versions, timing, belief fits
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

from src.backtest import BeliefSchedule, ReplaySettings, load_archive, replay, summarize, validate_archive
from src.config import config_hash, load_config, resolve
from src.evaluation import paired_differences, policy_means
from src.utility import UtilityParams

POLICY_ORDER = ("free_only", "always_buy", "fixed_order", "random", "myopic_voi", "optimal_voi")


def make_policies(cfg: dict, names=POLICY_ORDER) -> list:
    """Instantiate the six policies with canonical names (see src/policies.py)."""
    import src.policies as P

    d = cfg["decision"]
    n_quad = d.get("n_quad")
    factories = {
        "free_only": lambda: P.FreeOnly(),
        "always_buy": lambda: P.AlwaysBuy(),
        "fixed_order": lambda: P.FixedOrder(order=("A", "B")),
        "random": lambda: P.RandomAcquisition(p=float(d.get("random_policy_p", 0.5))),
        "myopic_voi": lambda: P.MyopicVOI(n_nodes=n_quad),
        "optimal_voi": lambda: P.OptimalVOI(n_nodes=n_quad),
    }
    out = []
    for name in names:
        pol = factories[name]()
        pol.name = name
        out.append(pol)
    return out


def settings_from_cfg(cfg: dict) -> ReplaySettings:
    d = cfg["decision"]
    return ReplaySettings(
        params=UtilityParams(lam=float(d["lam"]), tau=float(d["tau"])),
        costs={"A": float(d["cost_A"]), "B": float(d["cost_B"])},
        budget=float(d["budget"]),
        n_quad=d.get("n_quad"),
        seed=int(cfg["seed"]),
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=resolve("."), text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_overrides(cfg: dict, sets: list[str]) -> dict:
    import copy

    cfg = copy.deepcopy(cfg)
    for s in sets or []:
        key, val = s.split("=", 1)
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            node[parts[-1]] = json.loads(val)
        except json.JSONDecodeError:
            node[parts[-1]] = val
    return cfg


def prepare(cfg: dict, synthetic: int = 0, archive: pd.DataFrame | None = None):
    """Return (archive, eval_mask, schedule, data_info)."""
    info: dict = {}
    if archive is not None:
        archive = validate_archive(archive)
        info["source"] = "provided"
    elif synthetic:
        from src.synthetic_archive import make_synthetic_archive, spy_like_model

        archive = validate_archive(make_synthetic_archive(spy_like_model(), synthetic, seed=int(cfg["seed"])))
        info.update(source="SYNTHETIC", n=synthetic)
    else:
        path = resolve(cfg["forecast"]["archive_path"])
        archive = load_archive(path)
        info.update(source=str(path.relative_to(resolve("."))), sha256=_sha256(path))
    if synthetic or bool(archive.get("synthetic", pd.Series([False])).any()):
        # synthetic: calibrate on the first 60% of dates, evaluate on the rest
        cut = archive["date"].iloc[int(0.6 * len(archive))]
        cal_end, eval_start = cut - pd.Timedelta(days=1), cut
    else:
        cal_end = pd.Timestamp(cfg["periods"]["calibration_end"])
        eval_start = pd.Timestamp(cfg["periods"]["evaluation_start"])
    if eval_start <= cal_end:
        raise ValueError("evaluation must start after the calibration period ends")
    eval_mask = (archive["date"] >= eval_start).to_numpy()
    b = cfg["belief"]
    schedule = BeliefSchedule(archive=archive, calibration_end=cal_end, method=b["method"], mode=b["mode"],
                              refit_every=int(b.get("refit_every", 21)),
                              calibration_days=int(b.get("calibration_days", 0)))
    info.update(calibration_end=str(cal_end.date()), evaluation_start=str(eval_start.date()),
                n_archive=len(archive), n_eval=int(eval_mask.sum()),
                n_calibration_rows=int((archive["date"] <= cal_end).sum()),
                archive_first=str(archive["date"].iloc[0].date()), archive_last=str(archive["date"].iloc[-1].date()))
    return archive, eval_mask, schedule, info


def run(cfg: dict, tag: str = "default", synthetic: int = 0, max_dates: int = 0, archive: pd.DataFrame | None = None,
        policy_names=POLICY_ORDER, save: bool = True, progress_every: int = 250, n_boot: int = 2000):
    t0 = time.time()
    archive, eval_mask, schedule, info = prepare(cfg, synthetic=synthetic, archive=archive)
    if max_dates:
        idx = np.flatnonzero(eval_mask)[:max_dates]
        eval_mask = np.zeros_like(eval_mask)
        eval_mask[idx] = True
        info["n_eval"] = int(eval_mask.sum())
        info["max_dates"] = max_dates
    policies = make_policies(cfg, policy_names)
    settings = settings_from_cfg(cfg)
    log = replay(archive, eval_mask, schedule, policies, settings, progress_every=progress_every)
    summary = summarize(log)
    paired = paired_differences(log, baseline="free_only", n_boot=n_boot, seed=int(cfg["seed"])) \
        if "free_only" in policy_names else pd.DataFrame()
    means = policy_means(log, n_boot=n_boot, seed=int(cfg["seed"]))
    summary = summary.join(means.set_index("policy")[["ci_lo_bp", "ci_hi_bp"]].rename(
        columns={"ci_lo_bp": "net_ci_lo_bp", "ci_hi_bp": "net_ci_hi_bp"}))
    runtime = time.time() - t0
    m = schedule.model_for(archive.loc[eval_mask, "date"].iloc[0], 0)
    manifest = {
        "tag": tag, "config": cfg, "config_hash": config_hash(cfg), "seed": cfg["seed"], "data": info,
        "git_commit": _git_commit(), "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "runtime_seconds": runtime,
        "belief_fits": [{k: str(v) for k, v in f.items()} for f in schedule.fit_log[:50]],
        "n_belief_fits": len(schedule.fit_log),
        "first_belief_model": {"mean": m.mean.tolist(), "cov": m.cov.tolist(), "meta": {k: str(v) for k, v in m.meta.items()}},
        "nan_counts": {c: int(v) for c, v in log.isna().sum().items() if v},
    }
    if save:
        root = resolve("results")
        (root / "logs").mkdir(parents=True, exist_ok=True)
        (root / "tables").mkdir(parents=True, exist_ok=True)
        log.to_csv(root / "logs" / f"{tag}_log.csv.gz", index=False)
        summary.to_csv(root / "tables" / f"{tag}_summary.csv")
        if not paired.empty:
            paired.to_csv(root / "tables" / f"{tag}_paired.csv", index=False)
        with open(root / "tables" / f"{tag}_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, default=str)
    return log, summary, paired, manifest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.toml")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--synthetic", type=int, default=0, help="use a SYNTHETIC archive with this many dates")
    ap.add_argument("--max-dates", type=int, default=0, help="evaluate only the first K evaluation dates")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. decision.lam=5")
    args = ap.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.set)
    tag = args.tag or ("synthetic" if args.synthetic else "default")
    if args.synthetic:
        tag = tag if tag.startswith("synthetic") else f"synthetic_{tag}"
    log, summary, paired, manifest = run(cfg, tag=tag, synthetic=args.synthetic, max_dates=args.max_dates)
    pd.set_option("display.width", 200)
    print(f"\n== {tag}: {manifest['data']['n_eval']} evaluation dates, source={manifest['data']['source']}, "
          f"{manifest['runtime_seconds']:.1f}s ==")
    print(summary.round(4).to_string())
    if not paired.empty:
        print(paired.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
