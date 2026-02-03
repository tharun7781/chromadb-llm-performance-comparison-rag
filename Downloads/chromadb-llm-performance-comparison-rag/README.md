# ChromaDB LLM Performance Comparison (RAG)

This project compares LLMs across three questions on resume data.

Quick steps (sample 100 resumes):

1. Install dependencies:
   pip install -r requirements.txt

2. Prepare extracted resume texts:
   python scripts/prepare_resumes.py --data-dir app/Data --out data/resumes.csv

3. Run LLM comparison (local adapter for quick tests):
   python app/llm_compare.py --resumes data/resumes.csv --sample 100 --out results/results_sample_100.csv --llms local

4. Evaluate with heuristic gold labels:
   python app/evaluate.py --resumes data/resumes.csv --results results/results_sample_100.csv --out results/results_eval_sample_100.csv --summary results/summary_sample_100.json

5. View results in Streamlit:
   streamlit run app/dashboard.py

Notes:
- To include OpenAI / Anthropic providers, set `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` in your environment (or in a `.env` file) and use `--llms openai,anthropic,local`.
- The gold labels are heuristic (rule-based) extractions from resume text and are not human-verified; use them for rough accuracy estimates.
