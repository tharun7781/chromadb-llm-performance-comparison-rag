@echo off
echo Activating virtual environment...
call venv\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt

echo Running RAG pipeline...
python app/rag_query.py

echo Running LLM comparison...
python app/llm_compare.py

echo Launching dashboard...
streamlit run app/dashboard.py

pause
