# AI Debugging Copilot

## Overview
An AI-powered debugging assistant that identifies and explains Python code errors using retrieval-augmented generation (RAG).

## Features
- Detects common programming errors
- Provides Error, Reason, and Fix
- Uses semantic search (FAISS)
- Built with LangChain pipeline
- Interactive web UI using Streamlit

## Tech Stack
- Python
- LangChain
- FAISS
- Sentence Transformers
- Streamlit

## How It Works
1. Code input is converted into embeddings
2. FAISS retrieves similar error patterns
3. AI generates explanation
4. Fallback ensures reliable output

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py