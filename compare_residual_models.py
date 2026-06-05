from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

from dashboard_data import inventory_decomposition, load_full_inventory_data


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
SUMMARY_JSON = DOCS_DIR / "residual_model_comparison.json"


@dataclass(frozen=True)
class ModelResult:
    model_name: str
    aic: float
    bic: float
    innovation_count: int
    innovation_std_bcf: float
    acf_lag_1: float
    acf_lag_5: float
    max_abs_acf_1_13: float
    ljung_box_pvalue_13: float
    ljung_box_pvalue_26: float
    clean_score: float


def load_stl_residuals() -> pd.Series:
    inventory = load_full_inventory_data(BASE_DIR)
    decomp = inventory_decomposition(inventory)
    residual = pd.Series(decomp.resid, index=decomp.observed.index).dropna().astype(float)
    return residual


def fit_autoreg(residual: pd.Series, lags: int) -> tuple[pd.Series, float, float]:
    model = AutoReg(residual, lags=lags, old_names=False).fit()
    innovations = pd.Series(model.resid, index=residual.index[-len(model.resid):]).dropna()
    return innovations, float(model.aic), float(model.bic)


def fit_arima(residual: pd.Series, order: tuple[int, int, int]) -> tuple[pd.Series, float, float]:
    model = ARIMA(
        residual,
        order=order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()
    innovations = pd.Series(model.resid, index=residual.index).dropna()
    return innovations, float(model.aic), float(model.bic)


def evaluate_noise(model_name: str, innovations: pd.Series, aic: float, bic: float) -> ModelResult:
    clean = innovations.dropna().astype(float)
    acf_lag_1 = float(clean.autocorr(1))
    acf_lag_5 = float(clean.autocorr(5))
    lb = acorr_ljungbox(clean, lags=[13, 26], return_df=True)
    lag_autocorrs = [abs(float(clean.autocorr(lag))) for lag in range(1, 14)]
    max_abs_acf_1_13 = float(np.nanmax(lag_autocorrs))

    # Higher is better: reward higher Ljung-Box p-values and lower residual autocorrelation.
    clean_score = float(
        0.45 * lb.loc[13, "lb_pvalue"]
        + 0.45 * lb.loc[26, "lb_pvalue"]
        - 0.10 * max_abs_acf_1_13
    )

    return ModelResult(
        model_name=model_name,
        aic=aic,
        bic=bic,
        innovation_count=int(len(clean)),
        innovation_std_bcf=float(clean.std()),
        acf_lag_1=acf_lag_1,
        acf_lag_5=acf_lag_5,
        max_abs_acf_1_13=max_abs_acf_1_13,
        ljung_box_pvalue_13=float(lb.loc[13, "lb_pvalue"]),
        ljung_box_pvalue_26=float(lb.loc[26, "lb_pvalue"]),
        clean_score=clean_score,
    )


def build_summary(results: list[ModelResult], residual: pd.Series) -> dict[str, object]:
    ranked = sorted(results, key=lambda row: row.clean_score, reverse=True)
    best = ranked[0]
    return {
        "coverage_start": residual.index.min().strftime("%Y-%m-%d"),
        "coverage_end": residual.index.max().strftime("%Y-%m-%d"),
        "stl_residual_count": int(len(residual)),
        "ranking_basis": (
            "Higher clean_score is better. Score rewards higher Ljung-Box p-values at 13 and 26 lags "
            "and penalizes the largest absolute autocorrelation across lags 1-13."
        ),
        "best_model": best.model_name,
        "results": [asdict(result) for result in ranked],
    }


def main() -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    residual = load_stl_residuals()

    ar4_innovations, ar4_aic, ar4_bic = fit_autoreg(residual, lags=4)
    ar5_innovations, ar5_aic, ar5_bic = fit_autoreg(residual, lags=5)
    arima_innovations, arima_aic, arima_bic = fit_arima(residual, order=(4, 0, 1))

    results = [
        evaluate_noise("AR(4)", ar4_innovations, ar4_aic, ar4_bic),
        evaluate_noise("AR(5)", ar5_innovations, ar5_aic, ar5_bic),
        evaluate_noise("ARIMA(4,0,1)", arima_innovations, arima_aic, arima_bic),
    ]
    summary = build_summary(results, residual)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Coverage: {summary['coverage_start']} to {summary['coverage_end']}")
    print(f"Best model by residual whiteness: {summary['best_model']}")
    for result in summary["results"]:
        print(
            f"{result['model_name']}: "
            f"clean_score={result['clean_score']:.4f}, "
            f"acf(1)={result['acf_lag_1']:.4f}, "
            f"acf(5)={result['acf_lag_5']:.4f}, "
            f"max|acf|1-13={result['max_abs_acf_1_13']:.4f}, "
            f"LB13 p={result['ljung_box_pvalue_13']:.4f}, "
            f"LB26 p={result['ljung_box_pvalue_26']:.4f}, "
            f"AIC={result['aic']:.2f}"
        )
    print(f"Wrote {SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
