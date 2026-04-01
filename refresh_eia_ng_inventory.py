from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


SERIES_ID = "NG.NW2_EPG0_SWO_R48_BCF.W"
SERIES_CODE = "NW2_EPG0_SWO_R48_BCF"
BASE_URL = "https://api.eia.gov/v2/seriesid/NG.NW2_EPG0_SWO_R48_BCF.W"


@dataclass(frozen=True)
class Config:
    api_key: str
    years: int
    output_dir: Path


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Refresh the EIA weekly Lower 48 natural gas inventory history."
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("EIA_API_KEY"),
        help="EIA API key. Defaults to the EIA_API_KEY environment variable.",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=10,
        help="Number of years of history to keep. Default: 10.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where the JSON and CSV files will be written. Default: current directory.",
    )
    args = parser.parse_args()

    if not args.api_key:
        parser.error("Missing API key. Pass --api-key or set EIA_API_KEY.")
    if args.years <= 0:
        parser.error("--years must be a positive integer.")

    return Config(
        api_key=args.api_key,
        years=args.years,
        output_dir=Path(args.output_dir).resolve(),
    )


def fetch_series(api_key: str) -> dict:
    url = f"{BASE_URL}?api_key={api_key}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "nat-gas-research-refresh-script/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def trim_rows(rows: list[dict], years: int) -> list[dict]:
    cutoff = date.today().replace(year=date.today().year - years)
    filtered = [
        row
        for row in rows
        if datetime.strptime(row["period"], "%Y-%m-%d").date() >= cutoff
    ]
    return sorted(filtered, key=lambda row: row["period"])


def write_json(path: Path, rows: list[dict], years: int) -> None:
    payload = {
        "source": "EIA API v2",
        "series_id": SERIES_ID,
        "series_code": SERIES_CODE,
        "retrieved_at": datetime.now().isoformat(timespec="seconds"),
        "years_requested": years,
        "start_period": rows[0]["period"],
        "end_period": rows[-1]["period"],
        "row_count": len(rows),
        "units": rows[0]["units"],
        "data": rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["period", "value_bcf", "units", "series", "description"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "period": row["period"],
                    "value_bcf": row["value"],
                    "units": row["units"],
                    "series": row["series"],
                    "description": row["series-description"],
                }
            )


def main() -> int:
    config = parse_args()
    try:
        response = fetch_series(config.api_key)
        rows = trim_rows(response["response"]["data"], config.years)
    except urllib.error.HTTPError as exc:
        print(f"HTTP error from EIA API: {exc.code}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Network error while calling EIA API: {exc.reason}", file=sys.stderr)
        return 1
    except KeyError as exc:
        print(f"Unexpected API response format, missing key: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Failed to parse dates from API response: {exc}", file=sys.stderr)
        return 1

    if not rows:
        print("No rows returned for the requested time window.", file=sys.stderr)
        return 1

    config.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = config.output_dir / "eia_ng_total_inventory_last_10_years.json"
    csv_path = config.output_dir / "eia_ng_total_inventory_last_10_years.csv"

    write_json(json_path, rows, config.years)
    write_csv(csv_path, rows)

    print(f"Wrote {len(rows)} rows to {json_path}")
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Coverage: {rows[0]['period']} to {rows[-1]['period']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
