"""Prepare resumes.csv by extracting text from PDFs in app/Data

Usage:
    python scripts/prepare_resumes.py --data-dir app/Data --out data/resumes.csv

Dependencies: pypdf (pip install pypdf)
"""
import argparse
from pathlib import Path
import csv
import sys

try:
    from pypdf import PdfReader
except Exception as e:
    print("Error: pypdf is required. pip install pypdf", file=sys.stderr)
    raise


def extract_text_from_pdf(path: Path) -> str:
    text_parts = []
    try:
        reader = PdfReader(str(path))
        for p in reader.pages:
            try:
                text_parts.append(p.extract_text() or "")
            except Exception:
                continue
    except Exception as e:
        return ""
    return "\n".join(text_parts).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="app/Data")
    parser.add_argument("--out", default="data/resumes.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    files = sorted([p for p in data_dir.glob("*.pdf")])
    print(f"Found {len(files)} pdf files in {data_dir}")

    with out_path.open("w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["resume_id", "filename", "text"])
        writer.writeheader()
        for idx, p in enumerate(files, start=1):
            text = extract_text_from_pdf(p)
            if not text:
                print(f"Warning: no text extracted from {p.name}")
            writer.writerow({
                "resume_id": idx,
                "filename": p.name,
                "text": text,
            })
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
