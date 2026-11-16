"""Build the walk-forward forecast archive.

    python3 -m experiments.build_archive [--config configs/default.toml] [--n-jobs N]
                                         [--smoke] [--no-download]

Full build: writes data/processed/forecast_archive.csv and forecast_archive_manifest.json.
Smoke build (plumbing check on the last 1500 labelled rows): writes forecast_archive_smoke.csv
and forecast_archive_smoke_manifest.json with ``smoke: true``. The full build is meant for
GitHub Actions (``.github/workflows/pipeline.yml``), not for the development machine.

``--n-jobs`` parallelizes the refit blocks (joblib). The random forest runs single-threaded
inside each block.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_pipeline as dp  # noqa: E402
from src.forecast_archive import build_archive, load_config, write_archive  # noqa: E402


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.toml"))
    ap.add_argument("--n-jobs", type=int, default=1, help="joblib workers for refit blocks (-1 = all cores)")
    ap.add_argument("--smoke", action="store_true", help="tiny plumbing run; writes the *_smoke files")
    ap.add_argument("--no-download", action="store_true", help="use the existing raw CSV; do not call Yahoo")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    raw_path = _resolve(data_cfg["raw_path"])
    if args.no_download:
        if not raw_path.exists():
            raise SystemExit(f"--no-download given but {raw_path} does not exist")
    else:
        dp.download_prices(data_cfg["ticker"], data_cfg["start"], data_cfg["end"], raw_path)
    raw_sha = dp.file_sha256(raw_path)
    prices = dp.load_prices(raw_path)

    arc, manifest = build_archive(
        prices,
        cfg,
        n_jobs=args.n_jobs,
        smoke=args.smoke,
        raw_sha256=raw_sha,
    )
    out = _resolve(cfg["forecast"]["archive_path"])
    if args.smoke:
        out = out.with_name(f"{out.stem}_smoke{out.suffix}")
    write_archive(arc, manifest, out)

    counts = manifest["row_counts"]
    print(f"wrote {out} ({counts['archive_rows']} rows, {counts['refits']} refits, smoke={args.smoke})")
    print(f"origins {manifest['period']['first_origin']} .. {manifest['period']['last_origin']}")
    for key, value in manifest["diagnostics_not_for_selection"]["corr_forecast_with_realized_return"].items():
        print(f"  corr({key}_forecast, realized) = {value:+.4f}  [{manifest['diagnostics_not_for_selection']['label']}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
