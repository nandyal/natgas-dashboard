from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from dashboard_data import DEFAULT_TICKERS, fetch_market_prices, monthly_returns


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = BASE_DIR / "market_sentiment_events.csv"
OUTPUT_JSON = BASE_DIR / "market_sentiment_events.json"
RECENT_OUTPUT_CSV = BASE_DIR / "market_sentiment_recent.csv"
RECENT_OUTPUT_JSON = BASE_DIR / "market_sentiment_recent.json"
TICKERS = DEFAULT_TICKERS


def load_local_env() -> None:
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def get_vader() -> Any:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


def get_finbert() -> Any:
    from transformers import pipeline

    token = os.environ.get("FINBERT_API_KEY") or os.environ.get("HF_TOKEN")
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        truncation=True,
        token=token,
    )


def monthly_event_text(ticker: str, period: pd.Timestamp, monthly_ret: float, trailing_vol: float) -> str:
    direction = "rose" if monthly_ret >= 0 else "fell"
    strength = "sharply" if abs(monthly_ret) >= 0.10 else "moderately" if abs(monthly_ret) >= 0.05 else "slightly"
    return (
        f"For the month ending {period.strftime('%Y-%m-%d')}, {ticker} {strength} {direction} by "
        f"{abs(monthly_ret) * 100:.1f}%. The stock or fund showed trailing realized volatility of "
        f"{trailing_vol * 100:.1f}%, indicating {'elevated' if trailing_vol >= 0.25 else 'contained'} risk conditions "
        f"for natural gas-linked market exposure."
    )


def recent_event_text(
    ticker: str,
    as_of_date: pd.Timestamp,
    one_week_return: float,
    one_month_return: float,
    trailing_vol: float,
) -> str:
    week_direction = "rose" if one_week_return >= 0 else "fell"
    month_direction = "rose" if one_month_return >= 0 else "fell"
    return (
        f"As of {as_of_date.strftime('%Y-%m-%d')}, {ticker} {month_direction} by "
        f"{abs(one_month_return) * 100:.1f}% over the last 30 days and {week_direction} by "
        f"{abs(one_week_return) * 100:.1f}% over the last week. Realized volatility was "
        f"{trailing_vol * 100:.1f}%, indicating {'elevated' if trailing_vol >= 0.25 else 'contained'} risk conditions "
        f"for natural gas-linked market exposure."
    )


def latest_complete_month_end(index: pd.DatetimeIndex) -> pd.Timestamp:
    max_date = pd.Timestamp(index.max())
    current_month_end = max_date.to_period("M").to_timestamp("M")
    if max_date.normalize() < current_month_end.normalize():
        return (max_date.to_period("M") - 1).to_timestamp("M")
    return current_month_end


def build_market_sentiment_events() -> pd.DataFrame:
    close = fetch_market_prices(TICKERS, start="2019-01-01")
    monthly = monthly_returns(close)
    daily = close.pct_change().dropna(how="all")
    complete_through = latest_complete_month_end(close.index)
    monthly = monthly[monthly.index <= complete_through]
    next_month_return = monthly.shift(-1)

    records: list[dict] = []
    for ticker in close.columns:
        if ticker not in monthly.columns:
            continue
        monthly_series = monthly[ticker].dropna()
        trailing_vol = daily[ticker].rolling(63).std() * (252 ** 0.5)
        for period, monthly_ret in monthly_series.items():
            records.append(
                {
                    "ticker": ticker,
                    "period": period,
                    "monthly_return_pct": monthly_ret * 100,
                    "trailing_vol_pct": float(trailing_vol.asof(period) * 100) if pd.notna(trailing_vol.asof(period)) else None,
                    "forward_1m_return_pct": float(next_month_return[ticker].get(period) * 100) if pd.notna(next_month_return[ticker].get(period)) else None,
                    "event_text": monthly_event_text(
                        ticker=ticker,
                        period=period,
                        monthly_ret=float(monthly_ret),
                        trailing_vol=float(trailing_vol.asof(period)) if pd.notna(trailing_vol.asof(period)) else 0.0,
                    ),
                }
            )
    return pd.DataFrame(records)


def build_recent_market_sentiment(close: pd.DataFrame) -> pd.DataFrame:
    latest_date = pd.Timestamp(close.index.max())
    one_week_anchor = latest_date - pd.Timedelta(days=7)
    one_month_anchor = latest_date - pd.Timedelta(days=30)
    daily = close.pct_change().dropna(how="all")

    records: list[dict] = []
    for ticker in close.columns:
        series = close[ticker].dropna()
        if series.empty:
            continue
        latest_price = float(series.iloc[-1])
        week_anchor_price = float(series.asof(one_week_anchor))
        month_anchor_price = float(series.asof(one_month_anchor))
        if week_anchor_price == 0 or month_anchor_price == 0:
            continue
        one_week_return = latest_price / week_anchor_price - 1
        one_month_return = latest_price / month_anchor_price - 1
        trailing_vol = float((daily[ticker].dropna().tail(30).std() * (252 ** 0.5))) if ticker in daily.columns else 0.0
        records.append(
            {
                "ticker": ticker,
                "as_of_date": one_month_anchor.normalize(),
                "analysis_end_date": latest_date.normalize(),
                "one_week_return_pct": one_week_return * 100,
                "one_month_return_pct": one_month_return * 100,
                "trailing_vol_pct": trailing_vol * 100,
                "event_text": recent_event_text(
                    ticker=ticker,
                    as_of_date=latest_date,
                    one_week_return=float(one_week_return),
                    one_month_return=float(one_month_return),
                    trailing_vol=trailing_vol,
                ),
            }
        )
    return pd.DataFrame(records)


def main() -> int:
    load_local_env()
    close = fetch_market_prices(TICKERS, start="2019-01-01")
    events = build_market_sentiment_events()
    recent_events = build_recent_market_sentiment(close)
    vader = get_vader()
    finbert = get_finbert()

    vader_scores = events["event_text"].apply(vader.polarity_scores).apply(pd.Series)
    finbert_scores = events["event_text"].apply(lambda text: finbert(text)[0]).apply(pd.Series)

    events["vader_compound"] = vader_scores["compound"]
    events["vader_pos"] = vader_scores["pos"]
    events["vader_neu"] = vader_scores["neu"]
    events["vader_neg"] = vader_scores["neg"]
    events["finbert_label"] = finbert_scores["label"].str.lower()
    events["finbert_score"] = finbert_scores["score"]

    events = events.sort_values(["ticker", "period"], ascending=[True, False]).reset_index(drop=True)
    events.to_csv(OUTPUT_CSV, index=False)
    events.to_json(OUTPUT_JSON, orient="records", date_format="iso", indent=2)

    recent_vader_scores = recent_events["event_text"].apply(vader.polarity_scores).apply(pd.Series)
    recent_finbert_scores = recent_events["event_text"].apply(lambda text: finbert(text)[0]).apply(pd.Series)
    recent_events["vader_compound"] = recent_vader_scores["compound"]
    recent_events["vader_pos"] = recent_vader_scores["pos"]
    recent_events["vader_neu"] = recent_vader_scores["neu"]
    recent_events["vader_neg"] = recent_vader_scores["neg"]
    recent_events["finbert_label"] = recent_finbert_scores["label"].str.lower()
    recent_events["finbert_score"] = recent_finbert_scores["score"]
    recent_events = recent_events.sort_values("ticker").reset_index(drop=True)
    recent_events.to_csv(RECENT_OUTPUT_CSV, index=False)
    recent_events.to_json(RECENT_OUTPUT_JSON, orient="records", date_format="iso", indent=2)

    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {RECENT_OUTPUT_CSV}")
    print(f"Wrote {RECENT_OUTPUT_JSON}")
    print(f"Generated {len(events)} market sentiment events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
