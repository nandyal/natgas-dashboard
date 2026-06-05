from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import genpareto, norm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.seasonal import STL


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "eia_ng_total_inventory_full_history.csv"
DOCS_DIR = BASE_DIR / "docs"
MEP_PLOT = DOCS_DIR / "evt_mean_excess_plot.png"
TAIL_FIT_PLOT = DOCS_DIR / "evt_tail_fit_vs_gaussian.png"
SUMMARY_JSON = DOCS_DIR / "evt_model_summary.json"
THRESHOLD_BCF = 149.0


def load_inventory_series(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["period"]).sort_values("period")
    series = df.set_index("period")["value_bcf"].asfreq("W-FRI")
    return series.interpolate(limit_direction="both")


def stl_decompose(series: pd.Series):
    return STL(series, period=52, seasonal=53, trend=53, robust=True).fit()


def inventory_innovations(residual: pd.Series, ar_lags: int = 5) -> tuple[pd.Series, object]:
    clean = pd.Series(residual).dropna().astype(float)
    ar_model = AutoReg(clean, lags=ar_lags, old_names=False).fit()
    innovations = pd.Series(ar_model.resid, index=clean.index).dropna()
    return innovations, ar_model


def mean_excess_frame(shocks: pd.Series, threshold_to_mark: float) -> pd.DataFrame:
    clean = pd.Series(shocks).dropna().astype(float)
    lower = max(25.0, float(clean.quantile(0.70)))
    upper = float(clean.quantile(0.98))
    thresholds = np.unique(
        np.append(
            np.linspace(lower, upper, 35),
            threshold_to_mark,
        )
    )
    rows = []
    for threshold in thresholds:
        exceedances = clean[clean > threshold] - threshold
        rows.append(
            {
                "threshold": float(threshold),
                "mean_excess": float(exceedances.mean()) if len(exceedances) else np.nan,
                "n_exceedances": int(len(exceedances)),
            }
        )
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def save_mean_excess_plot(frame: pd.DataFrame, output_path: Path, threshold_to_mark: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    valid = frame.dropna(subset=["mean_excess"])
    ax.plot(valid["threshold"], valid["mean_excess"], color="#0f766e", linewidth=2.5)
    ax.scatter(valid["threshold"], valid["mean_excess"], color="#0f766e", s=18)
    ax.axvline(threshold_to_mark, color="#b45309", linestyle="--", linewidth=2, label=f"Test threshold = {threshold_to_mark:.0f} Bcf")
    target_row = valid.iloc[(valid["threshold"] - threshold_to_mark).abs().argsort()[:1]]
    if not target_row.empty:
        ax.scatter(target_row["threshold"], target_row["mean_excess"], color="#991b1b", s=50, zorder=3)
    ax.set_title("Mean Excess Plot for Inventory Innovations")
    ax.set_xlabel("Threshold u (Bcf)")
    ax.set_ylabel("Mean excess E[X - u | X > u] (Bcf)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fit_gpd(shocks: pd.Series, threshold: float) -> dict[str, float | int]:
    clean = pd.Series(shocks).dropna().astype(float)
    tail = clean[clean > threshold]
    exceedances = tail - threshold
    if len(exceedances) < 10:
        raise ValueError("Need at least 10 exceedances above the selected threshold for GPD fitting.")
    shape, _, scale = genpareto.fit(exceedances, floc=0)
    return {
        "threshold_bcf": float(threshold),
        "n_observations": int(len(clean)),
        "n_exceedances": int(len(exceedances)),
        "shape_xi": float(shape),
        "scale_sigma": float(scale),
        "tail_probability": float(len(exceedances) / len(clean)),
    }


def save_tail_fit_plot(
    shocks: pd.Series,
    threshold: float,
    shape: float,
    scale: float,
    innovation_std: float,
    output_path: Path,
) -> None:
    clean = pd.Series(shocks).dropna().astype(float)
    tail = clean[clean > threshold]
    folded_tail_prob = max(float(2 * norm.sf(threshold / innovation_std)), 1e-12) if innovation_std > 0 else np.nan
    x_values = np.linspace(threshold, max(float(clean.max()), threshold) * 1.05, 250)
    gpd_density = genpareto.pdf(x_values - threshold, shape, loc=0, scale=scale)
    gaussian_density = (
        2 * norm.pdf(x_values, loc=0, scale=innovation_std) / folded_tail_prob
        if innovation_std > 0 and np.isfinite(folded_tail_prob)
        else np.full_like(x_values, np.nan)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(tail, bins=18, density=True, alpha=0.55, color="#d97706", label="Observed tail shocks")
    ax.plot(x_values, gpd_density, color="#7c2d12", linewidth=2.8, label="Fitted GPD tail")
    ax.plot(x_values, gaussian_density, color="#1d4ed8", linewidth=2.2, linestyle="--", label="Gaussian tail expectation")
    ax.axvline(threshold, color="#92400e", linestyle="--", linewidth=2, label=f"POT threshold = {threshold:.0f} Bcf")
    ax.set_title("GPD Tail Fit vs Gaussian for Inventory Innovations")
    ax.set_xlabel("Absolute innovation shock (Bcf)")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    series = load_inventory_series(INPUT_CSV)
    stl_fit = stl_decompose(series)
    residual = pd.Series(stl_fit.resid, index=series.index).dropna()
    innovations, ar_model = inventory_innovations(residual, ar_lags=5)
    residual_shocks = residual.abs()

    mep = mean_excess_frame(residual_shocks, THRESHOLD_BCF)
    save_mean_excess_plot(mep, MEP_PLOT, THRESHOLD_BCF)

    fit = fit_gpd(residual_shocks, THRESHOLD_BCF)
    innovation_std = float(innovations.std())
    save_tail_fit_plot(
        shocks=residual_shocks,
        threshold=THRESHOLD_BCF,
        shape=float(fit["shape_xi"]),
        scale=float(fit["scale_sigma"]),
        innovation_std=float(residual.std()),
        output_path=TAIL_FIT_PLOT,
    )

    ljung_box = acorr_ljungbox(innovations, lags=[13, 26], return_df=True)
    summary = {
        "input_csv": str(INPUT_CSV),
        "coverage_start": series.index.min().strftime("%Y-%m-%d"),
        "coverage_end": series.index.max().strftime("%Y-%m-%d"),
        "stl_period_weeks": 52,
        "ar_lags": 5,
        "evt_fit_series": "absolute STL residual shocks",
        "threshold_bcf": THRESHOLD_BCF,
        "shape_xi": float(fit["shape_xi"]),
        "scale_sigma": float(fit["scale_sigma"]),
        "tail_probability": float(fit["tail_probability"]),
        "n_observations": int(fit["n_observations"]),
        "n_exceedances": int(fit["n_exceedances"]),
        "residual_std_bcf": float(residual.std()),
        "innovation_std_bcf": innovation_std,
        "innovation_acf_lag1": float(innovations.autocorr(1)),
        "ljung_box_pvalue_13": float(ljung_box.loc[13, "lb_pvalue"]),
        "ljung_box_pvalue_26": float(ljung_box.loc[26, "lb_pvalue"]),
        "innovation_threshold_exceedances_gt_149_bcf": int((innovations.abs() > THRESHOLD_BCF).sum()),
        "mean_excess_plot": str(MEP_PLOT),
        "tail_fit_plot": str(TAIL_FIT_PLOT),
        "ar_params": {str(name): float(value) for name, value in ar_model.params.items()},
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Coverage: {summary['coverage_start']} to {summary['coverage_end']}")
    print("EVT fit series: absolute STL residual shocks")
    print(f"Threshold tested: {THRESHOLD_BCF:.0f} Bcf")
    print(f"Shape xi: {summary['shape_xi']:.6f}")
    print(f"Scale sigma: {summary['scale_sigma']:.6f}")
    print(f"Exceedances: {summary['n_exceedances']} of {summary['n_observations']}")
    print(f"AR(5) innovation exceedances above 149 Bcf: {summary['innovation_threshold_exceedances_gt_149_bcf']}")
    print(f"Ljung-Box p-value (13 lags): {summary['ljung_box_pvalue_13']:.4f}")
    print(f"Ljung-Box p-value (26 lags): {summary['ljung_box_pvalue_26']:.4f}")
    print(f"Wrote {MEP_PLOT}")
    print(f"Wrote {TAIL_FIT_PLOT}")
    print(f"Wrote {SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
