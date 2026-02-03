"""LLM comparison runner (sample mode)

Usage:
  python app/llm_compare.py --resumes data/resumes.csv --sample 100 --out results/results_sample_100.csv --llms openai,anthropic,local

Notes:
- Requires OPENAI_API_KEY and/or ANTHROPIC_API_KEY in environment when using those providers.
- `local` adapter is a fast rule-based fallback for quick testing.
"""
import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
load_dotenv()

try:
    import openai
except Exception:
    openai = None

try:
    import anthropic
except Exception:
    anthropic = None


class BaseAdapter:
    def __init__(self, name: str):
        self.name = name

    def run(self, prompt: str, max_tokens: int = 256) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAIAdapter(BaseAdapter):
    def __init__(self, model: str = "gpt-3.5-turbo"):
        super().__init__("openai")
        self.model = model
        if openai is None:
            raise RuntimeError("openai package not installed")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def run(self, prompt: str, max_tokens: int = 256) -> Dict[str, Any]:
        start = time.time()
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        latency = (time.time() - start) * 1000
        text = resp.choices[0].message.content.strip()
        usage = resp.get("usage", {})
        return {
            "response": text,
            "latency_ms": latency,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }


class AnthropicAdapter(BaseAdapter):
    def __init__(self, model: str = "claude-2"):
        super().__init__("anthropic")
        self.model = model
        if anthropic is None:
            raise RuntimeError("anthropic package not installed")
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Client(api_key=key)

    def run(self, prompt: str, max_tokens: int = 256) -> Dict[str, Any]:
        start = time.time()
        resp = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens_to_sample=max_tokens,
            temperature=0,
        )
        latency = (time.time() - start) * 1000
        text = resp.completion.strip() if hasattr(resp, 'completion') else str(resp)
        return {
            "response": text,
            "latency_ms": latency,
            "prompt_tokens": None,
            "completion_tokens": None,
        }


class LocalRuleAdapter(BaseAdapter):
    """Simple rule-based answers for fast testing"""
    def __init__(self):
        super().__init__("local")

    def run(self, prompt: str, max_tokens: int = 256) -> Dict[str, Any]:
        start = time.time()
        p = prompt.lower()
        # very small heuristics
        if "highest degree" in p:
            for t in ["phd", "doctor", "ph.d"]:
                if t in p:
                    ans = "PhD"
                    break
            else:
                for t in ["master", "ms", "m.s"]:
                    if t in p:
                        ans = "Master's"
                        break
                else:
                    for t in ["bachelor", "ba", "bs", "b.s"]:
                        if t in p:
                            ans = "Bachelor's"
                            break
                    else:
                        ans = "Unknown"
        elif "years" in p:
            # count occurrences of "years" nearby numbers
            import re
            m = re.search(r"(\d+)\s+years", p)
            ans = m.group(1) if m else "Unknown"
        elif "python" in p:
            ans = "Yes" if "python" in p else "No"
        else:
            ans = "N/A"
        latency = (time.time() - start) * 1000
        return {"response": ans, "latency_ms": latency, "prompt_tokens": 0, "completion_tokens": 0}


QUESTIONS = [
    {
        "id": 1,
        "short": "highest_degree",
        "text": "What is the candidate's highest degree? Answer in one short phrase, e.g., PhD, Master's, Bachelor's, Associate, High School, Unknown.",
    },
    {
        "id": 2,
        "short": "years_experience",
        "text": "Approximately how many years of professional experience does the candidate have? Give a single integer or 'Unknown'.",
    },
    {
        "id": 3,
        "short": "has_python",
        "text": "Does the candidate list Python in skills or experience? Answer 'Yes' or 'No'.",
    },
]

PROMPT_TEMPLATE = """
You are given the following resume text delimited by triple backticks.
Please answer the question after it concisely.

Resume:
```{resume_text}```

Question: {question_text}

Answer:
"""


def load_resumes(path: Path) -> List[Dict[str, Any]]:
    import csv
    out = []
    with path.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            out.append(r)
    return out


def create_adapters(selected: List[str]):
    adapters = []
    for s in selected:
        if s == "openai":
            try:
                adapters.append(OpenAIAdapter())
            except Exception as e:
                print(f"Skipping openai adapter: {e}")
        elif s == "anthropic":
            try:
                adapters.append(AnthropicAdapter())
            except Exception as e:
                print(f"Skipping anthropic adapter: {e}")
        elif s == "local":
            adapters.append(LocalRuleAdapter())
        else:
            print(f"Unknown adapter {s}")
    return adapters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resumes", default="data/resumes.csv")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--out", default="results/results_sample_100.csv")
    parser.add_argument("--llms", default="local", help="comma-separated list: openai,anthropic,local")
    args = parser.parse_args()

    resumes_path = Path(args.resumes)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    resumes = load_resumes(resumes_path)
    sample = resumes[: args.sample]
    print(f"Loaded {len(resumes)} resumes, running on first {len(sample)} entries")

    adapters = create_adapters([s.strip() for s in args.llms.split(",") if s.strip()])
    if not adapters:
        print("No adapters available. Provide providers with API keys or use 'local' adapter.")
        return

    fieldnames = [
        "timestamp",
        "resume_id",
        "filename",
        "llm",
        "question_id",
        "question_short",
        "question",
        "response",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
    ]

    with out_path.open("w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in sample:
            resume_id = r.get("resume_id")
            fname = r.get("filename")
            resume_text = r.get("text", "")
            for q in QUESTIONS:
                question_text = q["text"]
                for adapter in adapters:
                    prompt = PROMPT_TEMPLATE.format(resume_text=resume_text[:8000], question_text=question_text)
                    try:
                        res = adapter.run(prompt)
                    except Exception as e:
                        print(f"Error running {adapter.name} on resume {resume_id}: {e}")
                        res = {"response": "ERROR", "latency_ms": None, "prompt_tokens": None, "completion_tokens": None}
                    row = {
                        "timestamp": time.time(),
                        "resume_id": resume_id,
                        "filename": fname,
                        "llm": adapter.name,
                        "question_id": q["id"],
                        "question_short": q["short"],
                        "question": question_text,
                        "response": res.get("response"),
                        "latency_ms": res.get("latency_ms"),
                        "prompt_tokens": res.get("prompt_tokens"),
                        "completion_tokens": res.get("completion_tokens"),
                    }
                    writer.writerow(row)
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
