"""
Stage 3a: load every per-station file in data/raw/ into ONE long DataFrame.

Output columns: timestamp | station | pm25
Saved to data/interim/pm25_long.csv

Run from the project root:  python src/load_data.py
"""
from pathlib import Path
import re
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/interim/pm25_long.csv")

# ---- Manual overrides: set these only if auto-detection picks wrong -------
TIMESTAMP_COL = None   # e.g. "From Date"
PM25_COL = None        # e.g. "PM2.5 (ug/m3)"
STATION_COL = None     # only if the station name is INSIDE the file
# ---------------------------------------------------------------------------


def norm(name: str) -> str:
    """'PM2.5 (ug/m3)' -> 'pm25ugm3' so matching ignores punctuation/case."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def find_pm25_col(df: pd.DataFrame, path: Path) -> str:
    if PM25_COL:
        return PM25_COL
    hits = [c for c in df.columns if "pm25" in norm(c)]
    if len(hits) != 1:
        raise ValueError(
            f"{path.name}: expected exactly one PM2.5 column, found {hits}. "
            f"Columns are: {list(df.columns)}. Set PM25_COL at the top."
        )
    return hits[0]


def build_timestamp(df: pd.DataFrame, path: Path) -> pd.Series:
    if TIMESTAMP_COL:
        return pd.to_datetime(df[TIMESTAMP_COL], errors="coerce", dayfirst=True)

    lower = {norm(c): c for c in df.columns}

    # Layout 1: separate year / month / day / hour columns
    if {"year", "month", "day", "hour"} <= lower.keys():
        parts = df[[lower["year"], lower["month"], lower["day"], lower["hour"]]]
        parts.columns = ["year", "month", "day", "hour"]
        return pd.to_datetime(parts, errors="coerce")

    # Layout 2: one datetime column, or a date column + a time column
    cands = [c for c in df.columns if any(k in norm(c) for k in ("date", "time"))]
    if len(cands) == 1:
        return pd.to_datetime(df[cands[0]], errors="coerce", dayfirst=True)
    if len(cands) == 2:
        joined = df[cands[0]].astype(str) + " " + df[cands[1]].astype(str)
        return pd.to_datetime(joined, errors="coerce", dayfirst=True)

    raise ValueError(
        f"{path.name}: can't identify the timestamp. Candidates: {cands}. "
        f"Columns are: {list(df.columns)}. Set TIMESTAMP_COL at the top."
    )


def load_one(path: Path) -> pd.DataFrame:
    df = read_any(path)
    pm_col = find_pm25_col(df, path)
    ts = build_timestamp(df, path)

    if STATION_COL:
        station = df[STATION_COL].astype(str)
    elif "station" in {norm(c) for c in df.columns}:
        station = df[[c for c in df.columns if norm(c) == "station"][0]].astype(str)
    else:
        station = path.stem  # fall back to the filename

    out = pd.DataFrame({
        "timestamp": ts,
        "station": station,
        "pm25": pd.to_numeric(df[pm_col], errors="coerce"),
    })

    bad_ts = out["timestamp"].isna().sum()
    print(f"  {path.name:<40} rows={len(out):>7}  pm25_col={pm_col!r}  bad_timestamps={bad_ts}")
    return out.dropna(subset=["timestamp"])


def main() -> None:
    files = sorted(p for p in RAW_DIR.iterdir()
                   if p.suffix.lower() in {".csv", ".xlsx", ".xls"})
    if not files:
        raise SystemExit(f"No .csv/.xlsx files found in {RAW_DIR.resolve()}")

    print(f"Found {len(files)} files:")
    frames = [load_one(p) for p in files]
    long_df = pd.concat(frames, ignore_index=True).sort_values(["station", "timestamp"])

    dupes = long_df.duplicated(["station", "timestamp"]).sum()
    print(f"\nCombined: {len(long_df):,} rows | duplicate (station, timestamp) pairs: {dupes}")

    summary = long_df.groupby("station").agg(
        rows=("pm25", "size"),
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        pct_missing=("pm25", lambda s: round(s.isna().mean() * 100, 1)),
        min_pm25=("pm25", "min"),
        max_pm25=("pm25", "max"),
    )
    print("\nPer-station summary:")
    print(summary.to_string())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()