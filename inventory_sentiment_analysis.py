from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dashboard_data import load_inventory_data


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = BASE_DIR / "inventory_sentiment_events.csv"
OUTPUT_JSON = BASE_DIR / "inventory_sentiment_events.json"


def build_event_text(row: pd.Series) -> str:
    direction = "drawdown" if row["weekly_change_bcf"] < 0 else "build"
    surprise = "unusually large" if row["abs_zscore"] >= 2 else "notable"
    relative_year = "above" if row["vs_year_ago_bcf"] >= 0 else "below"
    relative_avg = "above" if row["vs_52w_avg_bcf"] >= 0 else "below"
    return (
        f"For the week ending {row['period'].strftime('%Y-%m-%d')}, Lower 48 natural gas storage posted a "
        f"{surprise} {direction} of {abs(row['weekly_change_bcf']):.0f} Bcf, leaving inventories at "
        f"{row['value_bcf']:.0f} Bcf. Storage stood {abs(row['vs_year_ago_bcf']):.0f} Bcf {relative_year} "
        f"the level a year earlier and {abs(row['vs_52w_avg_bcf']):.0f} Bcf {relative_avg} the rolling 52-week average. "
        f"This suggests {'tighter' if row['weekly_change_bcf'] < 0 else 'looser'} near-term supply conditions in the natural gas market."
    )


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

    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        truncation=True,
    )


def classify_inventory_sentiment(change: float) -> str:
    return "bullish_gas" if change < 0 else "bearish_gas"


def main() -> int:
    df = load_inventory_data(BASE_DIR).copy()
    df["weekly_change_bcf"] = df["value_bcf"].diff()
    df["year_ago_bcf"] = df["value_bcf"].shift(52)
    df["vs_year_ago_bcf"] = df["value_bcf"] - df["year_ago_bcf"]
    df["vs_52w_avg_bcf"] = df["value_bcf"] - df["inventory_52w_avg"]
    df["change_mean_52w"] = df["weekly_change_bcf"].rolling(52).mean()
    df["change_std_52w"] = df["weekly_change_bcf"].rolling(52).std()
    df["zscore"] = (df["weekly_change_bcf"] - df["change_mean_52w"]) / df["change_std_52w"]
    df["abs_zscore"] = df["zscore"].abs()

    events = df[
        df["weekly_change_bcf"].notna()
        & df["abs_zscore"].notna()
        & (df["abs_zscore"] >= 1.5)
    ].copy()
    events["inventory_signal"] = events["weekly_change_bcf"].apply(classify_inventory_sentiment)
    events["event_text"] = events.apply(build_event_text, axis=1)

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

    columns = [
        "period",
        "value_bcf",
        "weekly_change_bcf",
        "vs_year_ago_bcf",
        "vs_52w_avg_bcf",
        "zscore",
        "abs_zscore",
        "inventory_signal",
        "vader_compound",
        "vader_pos",
        "vader_neu",
        "vader_neg",
        "finbert_label",
        "finbert_score",
        "event_text",
    ]
    events = events[columns].sort_values("period", ascending=False).reset_index(drop=True)
    events.to_csv(OUTPUT_CSV, index=False)
    events.to_json(OUTPUT_JSON, orient="records", date_format="iso", indent=2)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Detected {len(events)} unusual inventory events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
