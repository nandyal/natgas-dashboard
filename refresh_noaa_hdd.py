from __future__ import annotations

from pathlib import Path

import pandas as pd

from dashboard_data import (
    NOAA_HDD_CSV,
    NOAA_HDD_FULL_HISTORY_CSV,
    fetch_noaa_hdd_history,
    load_full_inventory_data,
    load_inventory_data,
)


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_HDD_ANOMALIES_CSV = BASE_DIR / "HDD data" / "processed_hdd_anomalies.csv"


def save_processed_hdd_anomalies(hdd_full: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "week_ending",
        "week_of_year",
        "us_hdd_weekly",
        "us_hdd_weekly_normal_30y",
        "us_hdd_weekly_std_30y",
        "us_hdd_weekly_anomaly_30y",
        "us_hdd_weekly_zscore_30y",
        "east_hdd_weekly",
        "east_hdd_weekly_normal_30y",
        "east_hdd_weekly_std_30y",
        "east_hdd_weekly_anomaly_30y",
        "east_hdd_weekly_zscore_30y",
        "midwest_hdd_weekly",
        "midwest_hdd_weekly_normal_30y",
        "midwest_hdd_weekly_std_30y",
        "midwest_hdd_weekly_anomaly_30y",
        "midwest_hdd_weekly_zscore_30y",
        "noaa_section_used",
    ]
    processed = hdd_full[columns].copy()
    processed = processed.rename(
        columns={
            "us_hdd_weekly_normal_30y": "us_hdd_30y_mean",
            "us_hdd_weekly_std_30y": "us_hdd_30y_std",
            "us_hdd_weekly_anomaly_30y": "hdd_anomaly",
            "us_hdd_weekly_zscore_30y": "hdd_anomaly_zscore",
            "east_hdd_weekly_normal_30y": "east_hdd_30y_mean",
            "east_hdd_weekly_std_30y": "east_hdd_30y_std",
            "east_hdd_weekly_anomaly_30y": "east_hdd_anomaly",
            "east_hdd_weekly_zscore_30y": "east_hdd_anomaly_zscore",
            "midwest_hdd_weekly_normal_30y": "midwest_hdd_30y_mean",
            "midwest_hdd_weekly_std_30y": "midwest_hdd_30y_std",
            "midwest_hdd_weekly_anomaly_30y": "midwest_hdd_anomaly",
            "midwest_hdd_weekly_zscore_30y": "midwest_hdd_anomaly_zscore",
        }
    )
    processed.to_csv(PROCESSED_HDD_ANOMALIES_CSV, index=False)
    return processed


def print_polar_vortex_verification(processed: pd.DataFrame) -> None:
    sample = processed.dropna(subset=["hdd_anomaly", "hdd_anomaly_zscore"]).copy()
    if sample.empty:
        print("Polar vortex verification skipped: no processed anomaly history available.")
        return
    target_date = pd.Timestamp("2014-01-10")
    target_rows = sample[sample["date"].eq(target_date)].copy()
    if target_rows.empty:
        winter_window = sample[(sample["date"] >= "2013-12-01") & (sample["date"] <= "2014-03-31")].copy()
        if winter_window.empty:
            print("Polar vortex verification skipped: no 2013-2014 winter rows found.")
            return
        target = winter_window.loc[winter_window["hdd_anomaly"].idxmax()]
    else:
        target = target_rows.iloc[0]
    if pd.isna(target["hdd_anomaly"]) or pd.isna(target["hdd_anomaly_zscore"]):
        print("Polar vortex verification skipped: no 2014 rows found.")
        return
    anomaly_percentile = float((sample["hdd_anomaly"] <= target["hdd_anomaly"]).mean() * 100)
    zscore_percentile = float((sample["hdd_anomaly_zscore"] <= target["hdd_anomaly_zscore"]).mean() * 100)
    print(
        "2014 Polar Vortex verification: "
        f"week ending {pd.Timestamp(target['date']).strftime('%Y-%m-%d')}, "
        f"HDD anomaly {target['hdd_anomaly']:+.2f}, "
        f"Z-score {target['hdd_anomaly_zscore']:+.2f}, "
        f"anomaly percentile {anomaly_percentile:.1f}th, "
        f"z-score percentile {zscore_percentile:.1f}th."
    )


def main() -> int:
    inventory = load_inventory_data(BASE_DIR)
    inventory_full = load_full_inventory_data(BASE_DIR)

    output_path = BASE_DIR / NOAA_HDD_CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hdd = fetch_noaa_hdd_history(inventory["period"])
    hdd.to_csv(output_path, index=False)
    print(f"Wrote {output_path} with {len(hdd)} weekly rows")

    full_output_path = BASE_DIR / NOAA_HDD_FULL_HISTORY_CSV
    hdd_full = fetch_noaa_hdd_history(inventory_full["period"])
    hdd_full.to_csv(full_output_path, index=False)
    print(f"Wrote {full_output_path} with {len(hdd_full)} weekly rows")

    processed = save_processed_hdd_anomalies(hdd_full)
    print(f"Wrote {PROCESSED_HDD_ANOMALIES_CSV} with {len(processed)} weekly rows")
    print_polar_vortex_verification(processed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
