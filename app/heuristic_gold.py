"""Heuristic extraction of gold labels from resume text for the three questions."""
import re
from pathlib import Path
from typing import Dict

DEGREE_KEYWORDS = [
    ("PhD", [r"\bph\.?d\b", r"doctor of", r"\bdr\b"]),
    ("Master's", [r"\bmaster\b", r"\bms\b", r"m\.s\b", r"m\.eng\b", r"m\.sc\b"]),
    ("Bachelor's", [r"\bbachelor\b", r"\bbs\b", r"b\.s\b", r"b\.a\b"]),
    ("Associate", [r"associate\b"]),
    ("High School", [r"high school\b"]),
]


def extract_highest_degree(text: str) -> str:
    t = (text or "").lower()
    for label, patterns in DEGREE_KEYWORDS:
        for p in patterns:
            if re.search(p, t):
                return label
    return "Unknown"


def extract_years_experience(text: str) -> str:
    t = (text or "")
    # find all occurrences like 'X years' or 'X+ years' or 'X years of experience'
    matches = re.findall(r"(\d{1,2})\+?\s+years", t, flags=re.IGNORECASE)
    if matches:
        # return the largest number found
        nums = [int(m) for m in matches]
        return str(max(nums))
    # also look for ranges like '3-5 years'
    m2 = re.findall(r"(\d{1,2})\s*[-–]\s*(\d{1,2})\s+years", t, flags=re.IGNORECASE)
    if m2:
        nums = [int(a) for a, b in m2] + [int(b) for a, b in m2]
        return str(max(nums))
    return "Unknown"


def extract_has_python(text: str) -> str:
    t = (text or "").lower()
    return "Yes" if "python" in t else "No"


def build_gold_map(resumes_csv_path: Path) -> Dict[str, Dict[str, str]]:
    import csv
    out = {}
    with resumes_csv_path.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rid = r.get("resume_id")
            text = r.get("text", "")
            out[rid] = {
                "highest_degree": extract_highest_degree(text),
                "years_experience": extract_years_experience(text),
                "has_python": extract_has_python(text),
            }
    return out


if __name__ == "__main__":
    import argparse
    import csv
    parser = argparse.ArgumentParser()
    parser.add_argument("--resumes", default="data/resumes.csv")
    parser.add_argument("--out", default="data/gold_sample_100.csv")
    args = parser.parse_args()

    gold = build_gold_map(Path(args.resumes))
    # write full file (all resumes)
    with open(args.out, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["resume_id", "highest_degree", "years_experience", "has_python"])
        writer.writeheader()
        for rid, vals in gold.items():
            row = {"resume_id": rid}
            row.update(vals)
            writer.writerow(row)
    print(f"Wrote {args.out}")