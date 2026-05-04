from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import genpareto
from statsmodels.tsa.seasonal import STL


BASE_DIR = Path(__file__).resolve().parent
INVENTORY_CSV = BASE_DIR / "eia_ng_total_inventory_full_history.csv"
HDD_CSV = BASE_DIR / "HDD data" / "processed_hdd_anomalies.csv"
DOCS_DIR = BASE_DIR / "docs"
PLOT_PATH = DOCS_DIR / "conditional_return_levels.png"
SUMMARY_PATH = DOCS_DIR / "conditional_risk_summary.json"


def load_inventory_residuals(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["period"]).sort_values("period")
    series = df.set_index("period")["value_bcf"].asfreq("W-FRI").interpolate(limit_direction="both")
    stl = STL(series, period=52, seasonal=53, trend=53, robust=True).fit()
    residual = pd.Series(stl.resid, index=series.index).dropna()
    result = pd.DataFrame(
        {
            "date": residual.index,
            "inventory_residual_bcf": residual.values,
        }
    )
    # Positive values here represent unexpected storage draws / tighter-than-seasonal conditions.
    result["draw_risk_bcf"] = (-result["inventory_residual_bcf"]).clip(lower=0)
    return result


def load_hdd_anomalies(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    needed = ["date", "week_of_year", "hdd_anomaly", "hdd_anomaly_zscore"]
    return df[needed].copy()


def merge_weather_and_residuals(residual_df: pd.DataFrame, hdd_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge_asof(
        residual_df.sort_values("date"),
        hdd_df.sort_values("date"),
        on="date",
        direction="nearest",
        tolerance=pd.Timedelta(days=3),
    )
    merged = merged.dropna(subset=["hdd_anomaly", "draw_risk_bcf"]).copy()
    merged["weather_tercile"] = pd.qcut(
        merged["hdd_anomaly"],
        q=3,
        labels=["Mild", "Normal", "Severe"],
        duplicates="drop",
    )
    return merged


def fit_conditional_gpd(
    values: pd.Series,
    threshold_quantile: float = 0.90,
    return_years: tuple[int, ...] = (10, 50, 100),
) -> dict[str, float | int]:
    clean = pd.Series(values).dropna().astype(float)
    clean = clean[clean > 0]
    if len(clean) < 60:
        raise ValueError("Need at least 60 positive draw-risk observations in the weather bucket.")

    threshold = float(clean.quantile(threshold_quantile))
    tail = clean[clean > threshold]
    exceedances = tail - threshold
    if len(exceedances) < 10:
        raise ValueError("Need at least 10 exceedances above the conditional threshold.")

    shape, _, scale = genpareto.fit(exceedances, floc=0)
    exceedance_rate = len(exceedances) / len(clean)

    metrics: dict[str, float | int] = {
        "n_obs": int(len(clean)),
        "n_exceedances": int(len(exceedances)),
        "threshold_bcf": float(threshold),
        "shape_xi": float(shape),
        "scale_sigma": float(scale),
        "mean_draw_risk_bcf": float(clean.mean()),
    }
    for years in return_years:
        target_probability = 1 / (52 * years)
        if abs(shape) < 1e-6:
            return_level = threshold + scale * np.log(exceedance_rate / target_probability)
        else:
            return_level = threshold + (scale / shape) * ((exceedance_rate / target_probability) ** shape - 1)
        metrics[f"return_level_{years}y_bcf"] = float(return_level)
    return metrics


def build_conditional_summary(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in ["Mild", "Normal", "Severe"]:
        bucket = merged.loc[merged["weather_tercile"] == label, "draw_risk_bcf"]
        metrics = fit_conditional_gpd(bucket)
        metrics["weather_tercile"] = label
        rows.append(metrics)
    return pd.DataFrame(rows)


def save_return_level_plot(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(summary))
    labels = summary["weather_tercile"].tolist()
    colors = {"10y": "#0f766e", "50y": "#1d4ed8", "100y": "#b45309"}

    for level in ["10y", "50y", "100y"]:
        values = summary[f"return_level_{level}_bcf"].to_numpy(dtype=float)
        ax.plot(x, values, marker="o", linewidth=2.5, color=colors[level], label=f"{level} return level")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Conditional Return Level Plot by Weather Tercile")
    ax.set_xlabel("Weather category from HDD anomaly")
    ax.set_ylabel("Storage draw risk (Bcf)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    residual_df = load_inventory_residuals(INVENTORY_CSV)
    hdd_df = load_hdd_anomalies(HDD_CSV)
    merged = merge_weather_and_residuals(residual_df, hdd_df)
    summary = build_conditional_summary(merged)
    save_return_level_plot(summary, PLOT_PATH)

    mild_xi = float(summary.loc[summary["weather_tercile"] == "Mild", "shape_xi"].iloc[0])
    severe_xi = float(summary.loc[summary["weather_tercile"] == "Severe", "shape_xi"].iloc[0])
    xi_delta = severe_xi - mild_xi

    payload = {
        "coverage_start": merged["date"].min().strftime("%Y-%m-%d"),
        "coverage_end": merged["date"].max().strftime("%Y-%m-%d"),
        "row_count": int(len(merged)),
        "mild_shape_xi": mild_xi,
        "severe_shape_xi": severe_xi,
        "xi_delta_severe_minus_mild": xi_delta,
        "summary": summary.to_dict(orient="records"),
        "plot_path": str(PLOT_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Merged sample: {payload['coverage_start']} to {payload['coverage_end']} ({payload['row_count']} rows)")
    print(f"Mild xi: {mild_xi:.6f}")
    print(f"Severe xi: {severe_xi:.6f}")
    print(f"Delta (Severe - Mild): {xi_delta:.6f}")
    print(summary[[
        'weather_tercile',
        'threshold_bcf',
        'shape_xi',
        'scale_sigma',
        'return_level_10y_bcf',
        'return_level_50y_bcf',
        'return_level_100y_bcf',
    ]].to_string(index=False))
    print(f"Wrote {PLOT_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
