from pathlib import Path
import json
import importlib.util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
LOG_PATH = REPO / "logs" / "app.log"
PROD_DIR = REPO / "cs-production"

# --- Load cslib from file path (works locally + in Docker) ---
CSLIB_PATH = REPO / "solution-guidance" / "cslib.py"
spec = importlib.util.spec_from_file_location("cslib", CSLIB_PATH)
cslib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cslib)

# --- NumPy patch (cslib may call np.in1d on newer NumPy) ---
if not hasattr(np, "in1d"):
    np.in1d = np.isin


def read_predict_events(log_path: Path) -> pd.DataFrame:
    """Read predict events from JSONL log file and return a tidy DataFrame."""
    if not log_path.exists():
        raise FileNotFoundError(f"No log file at {log_path}")

    rows = []
    for line in log_path.read_text().splitlines():
        try:
            event = json.loads(line)
        except Exception:
            continue

        if event.get("endpoint") != "predict":
            continue

        res = event.get("result", {}) or {}
        inp = event.get("payload", {}) or res.get("input", {}) or {}

        rows.append({
            "ts": event.get("ts"),
            "request_date": inp.get("date"),
            "country": inp.get("country", "all"),
            "used_row_date": res.get("used_row_date"),
            "prediction": res.get("prediction"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No predict events found in logs.")

    df["used_row_date"] = pd.to_datetime(df["used_row_date"], errors="coerce")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df = df.dropna(subset=["used_row_date", "prediction"]).sort_values("used_row_date")
    return df


def build_gold_target(prod_dir: Path, country: str = "all") -> pd.Series:
    """
    Build the gold target time series using the SAME definition as the model pipeline:
    cslib.engineer_features() returns (X, y, dates) where y is the target and dates align.
    """
    df_prod = cslib.fetch_data(str(prod_dir))
    ts_prod = cslib.convert_to_ts(df_prod, country=None if country in (None, "all") else country)

    # engineer_features returns: X (features), y (target), dates (date per row)
    X_gold, y_gold, dates_gold = cslib.engineer_features(ts_prod, training=False)

    gold = pd.Series(y_gold, index=pd.to_datetime(dates_gold, errors="coerce"), name="actual")
    gold = gold.dropna().sort_index()
    return gold


def main():
    preds = read_predict_events(LOG_PATH)

    # If you later support country-specific predictions, you can group by country here.
    # For now, treat everything as "all".
    gold_target = build_gold_target(PROD_DIR, country="all")

    # print("Gold target date range:", gold_target.index.min(), "to", gold_target.index.max())
    # print("Prediction used_row_date range:", preds["used_row_date"].min(), "to", preds["used_row_date"].max())

    # Align actual by used_row_date (the date used to generate the feature row)
    preds["actual"] = preds["used_row_date"].map(gold_target)

    eval_df = preds.dropna(subset=["actual"]).copy()
    if eval_df.empty:
        raise ValueError(
            "No overlapping dates between predictions and production gold target. "
            "This can happen if used_row_date is outside the gold target index."
        )

    # Metrics
    errors = eval_df["prediction"] - eval_df["actual"]
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    print(f"Evaluated predictions: {len(eval_df)}")
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")

    # Plot
    plt.figure()
    plt.plot(eval_df["used_row_date"], eval_df["actual"], label="Actual (gold target)")
    plt.plot(eval_df["used_row_date"], eval_df["prediction"], label="Predicted")
    plt.title("Post-production Monitoring: Predicted vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Target (next-30-day revenue sum)")
    plt.legend()
    plt.tight_layout()

    out_dir = REPO / "reports"
    out_dir.mkdir(exist_ok=True)

    fig_path = out_dir / "post_production_plot.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved plot: {fig_path}")

    csv_path = out_dir / "post_production_eval.csv"
    eval_df.to_csv(csv_path, index=False)
    print(f"Saved eval data: {csv_path}")


if __name__ == "__main__":
    main()