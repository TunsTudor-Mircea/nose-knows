"""
Lightweight reward model for NoseKnows.

Used when DPO pairs are insufficient (low feedback volume).
Trains a logistic-regression classifier on TF-IDF features of
(query + " [SEP] " + response) → score label (1=good, 0=bad).

The trained model can be used to:
  1. Filter synthetic training data — only keep triples that score above threshold.
  2. Serve as an additional inference-time guardrail alongside the existing guard.py.

Usage:
    # Train from Postgres feedback
    python -m src.rlhf.train_reward_model

    # Train and save to a custom path
    python -m src.rlhf.train_reward_model --output models/reward_model.pkl

    # Test the saved model on a sample input
    python -m src.rlhf.train_reward_model --test

Dependencies (all in base requirements.txt via scikit-learn):
    scikit-learn, joblib
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL_PATH = os.getenv("REWARD_MODEL_PATH", "models/reward_model.pkl")


async def _load_training_data() -> tuple[list[str], list[int]]:
    """Fetch (prompt+response, binary_label) pairs from Postgres feedback."""
    from sqlalchemy import select
    from src.db.session import AsyncSessionLocal
    from src.db.models import Feedback as DBFeedback, Message as DBMessage

    texts: list[str] = []
    labels: list[int] = []

    async with AsyncSessionLocal() as db:
        fb_rows = (
            await db.execute(
                select(DBFeedback).order_by(DBFeedback.created_at.asc())
            )
        ).scalars().all()

        for fb in fb_rows:
            asst_msg = (
                await db.execute(
                    select(DBMessage).where(DBMessage.id == fb.message_id)
                )
            ).scalar_one_or_none()

            if asst_msg is None or asst_msg.role != "assistant":
                continue

            user_msg = (
                await db.execute(
                    select(DBMessage)
                    .where(
                        DBMessage.session_id == asst_msg.session_id,
                        DBMessage.role == "user",
                        DBMessage.created_at < asst_msg.created_at,
                    )
                    .order_by(DBMessage.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()

            if user_msg is None:
                continue

            combined = f"{user_msg.content.strip()} [SEP] {asst_msg.content.strip()}"
            texts.append(combined)
            labels.append(1 if fb.score > 0 else 0)

    return texts, labels


def train(texts: list[str], labels: list[int], output_path: str) -> None:
    """Train a TF-IDF + logistic regression reward model and save it."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_val_score
        import joblib
    except ImportError:
        print("scikit-learn and joblib are required. Install them with: pip install scikit-learn joblib")
        sys.exit(1)

    if len(texts) < 4:
        print(f"Only {len(texts)} samples — need at least 4 to train. Collect more feedback first.")
        sys.exit(0)

    pos = sum(labels)
    neg = len(labels) - pos
    print(f"Training reward model on {len(texts)} samples ({pos} positive, {neg} negative)…")

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10_000, sublinear_tf=True)),
            ("clf", LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")),
        ]
    )

    if len(texts) >= 8:
        scores = cross_val_score(pipeline, texts, labels, cv=min(5, len(texts) // 2), scoring="roc_auc")
        print(f"  Cross-val ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}")

    pipeline.fit(texts, labels)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"  Model saved to: {output_path}")


def load_reward_model(model_path: str | None = None):
    """Load the saved reward model. Returns None if not found."""
    try:
        import joblib
    except ImportError:
        return None
    path = model_path or DEFAULT_MODEL_PATH
    if not Path(path).exists():
        return None
    return joblib.load(path)


def score_response(query: str, response: str, model=None) -> float:
    """
    Score a (query, response) pair using the reward model.
    Returns a probability in [0, 1] where 1 = likely good response.
    Returns 0.5 (neutral) if no model is available.
    """
    if model is None:
        model = load_reward_model()
    if model is None:
        return 0.5
    combined = f"{query.strip()} [SEP] {response.strip()}"
    proba = model.predict_proba([combined])[0]
    # Class 1 = positive/good
    classes = list(model.classes_)
    if 1 in classes:
        return float(proba[classes.index(1)])
    return float(proba[-1])


def _test_model(model_path: str) -> None:
    model = load_reward_model(model_path)
    if model is None:
        print(f"No model found at {model_path}. Train one first.")
        sys.exit(1)

    samples = [
        (
            "Something warm for a winter date night",
            "I recommend Chanel Coco Mademoiselle. It has warm amber base notes, rose heart, and a fresh citrus top that transitions beautifully to something cozy and romantic for the evening.",
        ),
        (
            "Something warm for a winter date night",
            "I don't know what you want. Perfumes smell nice.",
        ),
    ]

    print("Reward model scores:")
    for query, response in samples:
        s = score_response(query, response, model)
        print(f"  Score {s:.3f} — {response[:60]}…")


async def main(output: str, test: bool) -> None:
    if test:
        _test_model(output)
        return

    texts, labels = await _load_training_data()
    print(f"Loaded {len(texts)} labelled examples from Postgres.")

    if not texts:
        print("No feedback found. Rate some responses in the app first.")
        sys.exit(0)

    train(texts, labels, output)


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Train a lightweight reward model from NoseKnows user feedback."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to save the trained model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a quick sanity test on the saved model instead of training",
    )
    args = parser.parse_args()
    asyncio.run(main(args.output, args.test))


if __name__ == "__main__":
    cli()
