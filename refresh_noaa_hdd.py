from __future__ import annotations

from pathlib import Path

from dashboard_data import (
    NOAA_HDD_CSV,
    NOAA_HDD_FULL_HISTORY_CSV,
    fetch_noaa_hdd_history,
    load_full_inventory_data,
    load_inventory_data,
)


BASE_DIR = Path(__file__).resolve().parent


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
