#!/usr/bin/env python3
"""
Term Project Harness (prediction-only)

Usage
-----
python3 harness.py --input_csv <input file in csv> --output_csv <output csv file path>

Notes
-----
- The harness expects the trained artifacts file at the default path defined in estimation.py (ARTIFACTS_PATH).
- The output is a single-column CSV of PD estimates without a header and without index.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Optional
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

# Use predict.py only
from predict import predict_pd

# estimation module provides artifacts IO + utilities
import estimation
ModelArtifacts = estimation.ModelArtifacts
ARTIFACTS_DEFAULT = estimation.ARTIFACTS_PATH  # keep default if you like
engineer_financial_ratios = estimation.engineer_financial_ratios


def read_csv_safely(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        df = pd.read_csv(path, low_memory=False)
        if df.empty:
            raise ValueError(f"CSV is empty: {path}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV '{path}': {e}") from e


def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def save_series_as_csv(s: pd.Series, path: str, include_index: bool = False, header: bool = False) -> None:
    ensure_dir_for_file(path)
    s.to_csv(path, index=include_index, header=header)
    print(f"Wrote: {path}  (rows={len(s):,})")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Predict calibrated PDs with saved artifacts (submission harness)."
    )
    p.add_argument("--input_csv", required=True, help="CSV to score (test / OOT data).")
    p.add_argument(
        "--output_csv",
        required=True,
        help="Where to write calibrated PDs (single-column CSV, no header).",
    )
    return p


def do_predict(
    input_csv: str,
    output_csv: str,
) -> None:
    print(f"Loading scoring data from: {input_csv}")
    df_new = read_csv_safely(input_csv)
    print(f"Rows: {len(df_new):,}  Cols: {len(df_new.columns):,}")

    artifacts_path = ARTIFACTS_DEFAULT
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(
            f"Artifacts not found at '{artifacts_path}'. Ensure the artifacts file is included in your submission."
        )

    print(f"Loading artifacts from: {artifacts_path}")
    artifacts = estimation.load_artifacts(artifacts_path)

    print("Scoring calibrated PDs via predict.predict_pd(...)")
    pd_hat = predict_pd(df_new, artifacts=artifacts)

    print(f"Writing calibrated PDs to: {output_csv}")
    # Per spec: single column, no header, no index
    save_series_as_csv(pd_hat, output_csv, include_index=False, header=False)

    print("Prediction complete.")


def main(argv: Optional[list[str]] = None) -> int:
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 80)

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        do_predict(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
        )
        return 0
    except Exception as e:
        print("\nERROR:", str(e), file=sys.stderr)
        traceback.print_exc(limit=5)
        return 1


if __name__ == "__main__":
    sys.exit(main())
