"""
Data ingestion pipeline.

Reads dataset/fra_cleaned.csv, normalises all fields, builds one text chunk
per perfume, and writes the result to data/chunks.jsonl.

Usage:
    python -m src.data_pipeline
    python -m src.data_pipeline --input dataset/fra_cleaned.csv --output data/chunks.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fix_decimal(value: str | float) -> float | None:
    """Convert European comma-decimal strings (e.g. '3,42') to float."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    # Replace comma used as decimal separator only when it looks like X,XX
    s = re.sub(r"^(\d+),(\d+)$", r"\1.\2", s)
    try:
        return float(s)
    except ValueError:
        return None


def _parse_notes(raw: str | float) -> list[str]:
    """Parse a comma-separated notes string into a clean list."""
    if pd.isna(raw) or str(raw).strip() == "":
        return []
    return [n.strip() for n in str(raw).split(",") if n.strip()]


def _parse_accords(row: pd.Series) -> list[str]:
    """Collect up to 5 accord columns from a cleaned-CSV row."""
    accords = []
    for col in ["mainaccord1", "mainaccord2", "mainaccord3", "mainaccord4", "mainaccord5"]:
        val = row.get(col, "")
        if not pd.isna(val) and str(val).strip():
            accords.append(str(val).strip())
    return accords


def _build_chunk_text(row: pd.Series) -> str:
    """Format a single perfume row as a self-contained text chunk."""
    perfume = str(row.get("Perfume", "Unknown")).strip()
    brand = str(row.get("Brand", "Unknown")).strip()

    top = _parse_notes(row.get("Top", ""))
    middle = _parse_notes(row.get("Middle", ""))
    base = _parse_notes(row.get("Base", ""))
    accords = _parse_accords(row)

    lines = [f"{perfume} — {brand}"]
    if top:
        lines.append(f"Top notes: {', '.join(top)}")
    if middle:
        lines.append(f"Heart notes: {', '.join(middle)}")
    if base:
        lines.append(f"Base notes: {', '.join(base)}")
    if accords:
        lines.append(f"Accords: {', '.join(accords)}")

    gender = str(row.get("Gender", "")).strip()
    if gender:
        lines.append(f"Gender: {gender}")

    year = row.get("Year", "")
    if not pd.isna(year) and str(year).strip():
        lines.append(f"Year: {str(year).strip()}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_and_clean(csv_path: str | Path) -> pd.DataFrame:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(csv_path, sep=";", low_memory=False, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        # Last resort: replace undecodable bytes rather than crashing
        df = pd.read_csv(csv_path, sep=";", low_memory=False, encoding="utf-8", encoding_errors="replace")

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # Remove duplicate entries by url (keep first occurrence)
    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url"], keep="first")

    # Fix numeric fields with European decimal format
    df["Rating Value"] = df["Rating Value"].apply(_fix_decimal)
    df["Rating Count"] = df["Rating Count"].apply(_fix_decimal)

    # Drop rows with no perfume name
    df = df[df["Perfume"].notna() & (df["Perfume"].str.strip() != "")]
    df = df.reset_index(drop=True)

    return df


def build_chunks(df: pd.DataFrame) -> list[dict]:
    chunks = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building chunks"):
        text = _build_chunk_text(row)
        chunk = {
            "id": str(row.get("url", f"row_{_}")),
            "text": text,
            "metadata": {
                "perfume": str(row.get("Perfume", "")).strip(),
                "brand": str(row.get("Brand", "")).strip(),
                "gender": str(row.get("Gender", "")).strip(),
                "rating": row.get("Rating Value"),
                "rating_count": row.get("Rating Count"),
                "year": str(row.get("Year", "")).strip(),
                "top_notes": _parse_notes(row.get("Top", "")),
                "heart_notes": _parse_notes(row.get("Middle", "")),
                "base_notes": _parse_notes(row.get("Base", "")),
                "accords": _parse_accords(row),
                "url": str(row.get("url", "")).strip(),
            },
        }
        chunks.append(chunk)
    return chunks


def write_chunks(chunks: list[dict], output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Wrote {len(chunks):,} chunks → {output_path}")


def run(
    input_path: str = "dataset/fra_cleaned.csv",
    output_path: str = "data/chunks.jsonl",
) -> list[dict]:
    print(f"Loading dataset from {input_path} …")
    df = load_and_clean(input_path)
    print(f"  {len(df):,} rows after deduplication / cleaning")

    chunks = build_chunks(df)
    write_chunks(chunks, output_path)
    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NoseKnows data pipeline")
    parser.add_argument("--input", default="dataset/fra_cleaned.csv")
    parser.add_argument("--output", default="data/chunks.jsonl")
    args = parser.parse_args()
    run(args.input, args.output)
