"""Streamlit dashboard to view LLM comparison results and evaluation summary."""
import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="LLM Comparison Dashboard", layout="wide")

st.title("LLM Performance Comparison — 3 Questions 📊")

results_path = st.sidebar.text_input("Results CSV", "results/results_sample_100.csv")
summary_path = st.sidebar.text_input("Summary JSON", "results/summary_sample_100.json")

if not Path(results_path).exists():
    st.warning(f"Results file not found: {results_path}. Run the pipeline first.")
else:
    df = pd.read_csv(results_path)

    st.sidebar.markdown("---")
    llms = sorted(df['llm'].unique())
    qshorts = sorted(df['question_short'].unique())

    sel_llm = st.sidebar.selectbox("LLM", llms)
    sel_q = st.sidebar.selectbox("Question", qshorts)

    st.header(f"Summary for {sel_llm} — {sel_q}")

    # show aggregated metrics
    if Path(summary_path).exists():
        with open(summary_path) as f:
            summary = json.load(f)
        llm_summary = summary.get(sel_llm, {}).get(sel_q, {})
        st.metric("Accuracy", llm_summary.get('accuracy'))
        st.metric("Avg latency (ms)", llm_summary.get('avg_latency_ms'))
    else:
        st.info("Run evaluate.py to generate a summary JSON for dashboard metrics.")

    subset = df[(df['llm'] == sel_llm) & (df['question_short'] == sel_q)]

    st.subheader("Sample Responses")
    # show a few sample rows with filename, response, gold, correct
    cols = ['resume_id', 'filename', 'response']
    if 'gold' in subset.columns:
        cols += ['gold', 'correct']
    st.dataframe(subset[cols].head(200))

    st.subheader("Distributions")
    st.write(subset['response'].value_counts().head(20))

    st.sidebar.markdown("---")
    st.sidebar.markdown("Run locally: streamlit run app/dashboard.py")
