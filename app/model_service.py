from pathlib import Path
import joblib
import numpy as np
import sys

REPO = Path(__file__).resolve().parents[1]
from pathlib import Path
import importlib.util

REPO = Path(__file__).resolve().parents[1]

CSLIB_PATH = REPO / "solution-guidance" / "cslib.py"
spec = importlib.util.spec_from_file_location("cslib", CSLIB_PATH)
cslib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cslib)  # noqa

from src.modeling import LastValueRegressor

MODEL_PATH = REPO / "models" / "capstone_model.joblib"

def train_model(train_dir: Path) -> dict:
    df = cslib.fetch_data(str(train_dir))
    ts = cslib.convert_to_ts(df, country=None)

    # Patch for numpy if needed
    import numpy as np
    if not hasattr(np, "in1d"):
        np.in1d = np.isin

    X, y, dates = cslib.engineer_features(ts, training=True)

    model = LastValueRegressor().fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "rows": int(len(X)),
        "last_value": float(model.last_)
    }

def predict_next(model_input: dict, data_dir: Path) -> dict:
    """
    Input example:
      { "country": "all", "date": "2019-09-15" }

    Loads model, builds features from data_dir, selects the most recent row
    at or before the requested date, and predicts using that row.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not trained yet. Call /train first.")

    model = joblib.load(MODEL_PATH)

    # load + aggregate data
    df = cslib.fetch_data(str(data_dir))

    country = model_input.get("country", "all")
    ts = cslib.convert_to_ts(df, country=None if country in (None, "all") else country)

    # numpy patch (needed for cslib on newer numpy)
    import numpy as np
    if not hasattr(np, "in1d"):
        np.in1d = np.isin

    # build features
    X, y, dates = cslib.engineer_features(ts, training=False)

    # parse requested date
    import pandas as pd
    req_date = pd.to_datetime(model_input.get("date"))
    dates_pd = pd.to_datetime(dates)

    # choose latest available date <= requested date
    eligible_idx = np.where(dates_pd <= req_date)[0]
    if len(eligible_idx) == 0:
        raise ValueError(
            f"Requested date {req_date.date()} is earlier than available data "
            f"(starts at {dates_pd.min().date()})."
        )

    use_i = int(eligible_idx[-1])
    X_row = X.iloc[[use_i]]

    pred = float(model.predict(X_row)[0])

    return {
        "status": "ok",
        "prediction": pred,
        "used_row_date": str(dates_pd[use_i].date()),
        "input": model_input
    }