from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


SERIES_ID = "NG.NW2_EPG0_SWO_R48_BCF.W"
SERIES_CODE = "NW2_EPG0_SWO_R48_BCF"
BASE_URL = "https://api.eia.gov/v2/seriesid/NG.NW2_EPG0_SWO_R48_BCF.W"
ARCHIVE_URL = "https://www.eia.gov/dnav/ng/hist/nw_epg0_sao_r48_bcfw.htm"


@dataclass(frozen=True)
class Config:
    api_key: str
    years: int
    output_dir: Path
    full_history: bool


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
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Write the full available EIA series to eia_ng_total_inventory_full_history.* for EVT stress testing.",
    )
    args = parser.parse_args()

    if not args.api_key:
        parser.error("Missing API key. Pass --api-key or set EIA_API_KEY.")
    if not args.full_history and args.years <= 0:
        parser.error("--years must be a positive integer.")

    return Config(
        api_key=args.api_key,
        years=args.years,
        output_dir=Path(args.output_dir).resolve(),
        full_history=args.full_history,
    )


def fetch_series(api_key: str) -> dict:
    url = f"{BASE_URL}?api_key={api_key}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "nat-gas-research-refresh-script/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_legacy_archive_rows() -> list[dict]:
    request = urllib.request.Request(
        ARCHIVE_URL,
        headers={"User-Agent": "nat-gas-research-refresh-script/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        page = response.read().decode("latin1")
    rows: list[dict] = []
    for raw_row in re.findall(r"<tr>(.*?)</tr>", page, flags=re.IGNORECASE | re.DOTALL):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", raw_row, flags=re.IGNORECASE | re.DOTALL)
        if not cells:
            continue
        cleaned = [
            html.unescape(re.sub(r"<[^>]+>", "", cell))
            .replace("\xa0", " ")
            .strip()
            for cell in cells
        ]
        if not cleaned or not re.match(r"^\d{4}-[A-Za-z]{3}$", cleaned[0]):
            continue
        year = int(cleaned[0].split("-", 1)[0])
        for date_text, value_text in zip(cleaned[1::2], cleaned[2::2]):
            if not date_text or not value_text:
                continue
            value = value_text.replace(",", "").strip()
            if not value or not value.lstrip("-").isdigit():
                continue
            period = datetime.strptime(f"{year}/{date_text}", "%Y/%m/%d").strftime("%Y-%m-%d")
            rows.append(
                {
                    "period": period,
                    "value": int(value),
                    "units": "BCF",
                    "series": SERIES_CODE,
                    "series-description": "Weekly Lower 48 States Natural Gas Working Underground Storage (Billion Cubic Feet)",
                }
            )
    return sorted(rows, key=lambda row: row["period"])


def trim_rows(rows: list[dict], years: int) -> list[dict]:
    cutoff = date.today().replace(year=date.today().year - years)
    filtered = [
        row
        for row in rows
        if datetime.strptime(row["period"], "%Y-%m-%d").date() >= cutoff
    ]
    return sorted(filtered, key=lambda row: row["period"])


def merge_legacy_and_current_rows(legacy_rows: list[dict], current_rows: list[dict]) -> list[dict]:
    if not current_rows:
        return legacy_rows
    current_start = min(datetime.strptime(row["period"], "%Y-%m-%d").date() for row in current_rows)
    combined = [
        row
        for row in legacy_rows
        if datetime.strptime(row["period"], "%Y-%m-%d").date() < current_start
    ]
    combined.extend(current_rows)
    return sorted(combined, key=lambda row: row["period"])


def write_json(path: Path, rows: list[dict], years: int, full_history: bool) -> None:
    payload = {
        "source": "EIA API v2",
        "series_id": SERIES_ID,
        "series_code": SERIES_CODE,
        "retrieved_at": datetime.now().isoformat(timespec="seconds"),
        "years_requested": years,
        "full_history": full_history,
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
        api_rows = response["response"]["data"]
        rows = (
            merge_legacy_and_current_rows(fetch_legacy_archive_rows(), sorted(api_rows, key=lambda row: row["period"]))
            if config.full_history
            else trim_rows(api_rows, config.years)
        )
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
    if config.full_history:
        json_path = config.output_dir / "eia_ng_total_inventory_full_history.json"
        csv_path = config.output_dir / "eia_ng_total_inventory_full_history.csv"
    else:
        json_path = config.output_dir / "eia_ng_total_inventory_last_10_years.json"
        csv_path = config.output_dir / "eia_ng_total_inventory_last_10_years.csv"

    write_json(json_path, rows, config.years, config.full_history)
    write_csv(csv_path, rows)

    print(f"Wrote {len(rows)} rows to {json_path}")
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Coverage: {rows[0]['period']} to {rows[-1]['period']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
