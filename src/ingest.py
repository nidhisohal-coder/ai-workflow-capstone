# src/ingest.py
from pathlib import Path
import pandas as pd

def load_raw(data_dir: Path) -> pd.DataFrame:
    """
    Load JSON invoice files using the provided cslib.fetch_data() function.
    """
    import sys
    # make solution-guidance importable no matter where notebook is
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root / "solution-guidance"))

    import cslib  # noqa

    return cslib.fetch_data(str(data_dir))

def to_timeseries(df: pd.DataFrame, country: str | None = None) -> pd.DataFrame:
    """
    Convert raw invoice dataframe into daily time series using cslib.convert_to_ts().
    """
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root / "solution-guidance"))

    import cslib  # noqa

    return cslib.convert_to_ts(df, country=country)

def make_features(df_ts: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """
    Engineer lag/rolling features + target using cslib.engineer_features().
    """
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root / "solution-guidance"))

    import cslib  # noqa

    return cslib.engineer_features(df_ts, training=training)