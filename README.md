#  AI Debugging Copilot

> An AI-powered Python debugging tool that automatically detects bugs, explains the root cause, and suggests fixes — built using a RAG (Retrieval Augmented Generation) pipeline with LangChain, FAISS, and HuggingFace Transformers.

---

##  Problem Statement

Debugging is one of the most time-consuming parts of software development. Developers often spend hours on bugs that could be identified in seconds with the right context. This tool automates the first-pass analysis — acting like a senior developer who instantly tells you **what went wrong, why it happened, and how to fix it.**

---

##  Architecture

```
User pastes buggy Python code
            ↓
    ┌───────────────────┐
    │  Layer 1: exec()  │  ← Actually runs the code, catches real exception
    └───────────────────┘
            ↓
    ┌──────────────────────────────────────┐
    │  Layer 2: RAG Pipeline               │
    │  FAISS retrieves similar error       │  ← Semantic search on knowledge base
    │  → injected as context into prompt   │
    │  → distilgpt2 generates fix          │  ← LLM generates structured output
    └──────────────────────────────────────┘
            ↓
    ┌─────────────────────────┐
    │  Layer 3: Rule-Based    │  ← Fallback — guarantees structured response
    │  Fallback               │      even if LLM underperforms
    └─────────────────────────┘
            ↓
    Streamlit UI displays:
    🔴 Error | 🟡 Reason | 🟢 Fix
```

---

##  What is RAG? (Core Concept)

**RAG = Retrieval Augmented Generation**

Instead of asking the LLM to answer purely from its training data (which causes hallucination), RAG:

1. **Retrieves** the most semantically similar known error from FAISS vector store
2. **Augments** the LLM prompt with that retrieved context
3. **Generates** a fix that is grounded in real error patterns — not guessed

```
Without RAG:  Code → LLM → (possibly hallucinated fix)
With RAG:     Code → FAISS retrieval → Code + Context → LLM → (grounded fix)
```

This is the same architecture used in production AI systems at companies like Google, Meta, and Microsoft.

---

##  Tech Stack

| Technology | Version | Purpose | Why This Choice |
|---|---|---|---|
| **Streamlit** | ≥1.28 | Web UI | Build full web app in pure Python — no HTML/CSS needed |
| **LangChain** | ≥0.1.0 | RAG framework | Provides retriever, vector store, and chain abstractions |
| **FAISS** | ≥1.7.4 | Vector similarity search | Finds semantically similar errors using cosine distance — not keyword matching |
| **sentence-transformers/all-MiniLM-L6-v2** | — | Text embeddings | Converts text to 384-dimensional vectors for FAISS indexing |
| **distilgpt2** | — | Text generation | Lightweight offline LLM — no API key or cost required |
| **HuggingFace Transformers** | ≥4.35 | Model loading | High-level API to load and run AI models in one line |
| **pytest** | ≥7.4 | Unit testing | Industry-standard Python testing framework |

---

##  Supported Error Types

| Error | Example Code | What Tool Returns |
|---|---|---|
| `IndexError` | `arr = []; print(arr[0])` | List index out of range explanation + fix |
| `ZeroDivisionError` | `x = 10/0` | Division by zero explanation + fix |
| `TypeError` | `'hello' + 5` | Type mismatch explanation + fix |
| `SyntaxError` | `def foo(:` | Syntax issue explanation + fix |
| `NameError` | `print(x)` (x undefined) | Undefined variable explanation + fix |
| `KeyError` | `d['missing_key']` | Missing key explanation + fix |
| `AttributeError` | `(5).append(1)` | Invalid attribute explanation + fix |
| `ValueError` | `int('abc')` | Invalid value explanation + fix |

---

##  How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Diyamehta25/AI-Debugging-Copilot.git
cd AI-Debugging-Copilot

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py

# 4. Run the test suite
pytest test_debugger.py -v
```

---

##  Test Suite

The project includes **16 unit tests** across 4 test classes:

```
pytest test_debugger.py -v

TestValidCode
  PASSED  test_simple_print_no_error
  PASSED  test_list_operations_no_error

TestErrorDetection
  PASSED  test_index_error_detected
  PASSED  test_zero_division_detected
  PASSED  test_type_error_detected
  PASSED  test_name_error_detected
  PASSED  test_key_error_detected
  PASSED  test_attribute_error_detected

TestOutputStructure
  PASSED  test_result_contains_error_label
  PASSED  test_result_contains_reason_label
  PASSED  test_result_contains_fix_label
  PASSED  test_result_is_string

TestEdgeCases
  PASSED  test_empty_string_input
  PASSED  test_whitespace_only_input
  PASSED  test_multiline_code
  PASSED  test_none_not_returned

16 passed in Xs
```

---

##  Project Structure

```
AI-Debugging-Copilot/
│
├── app.py              # Streamlit UI layer — presentation only
├── debugger.py         # Core business logic — 3-layer debug pipeline
├── test_debugger.py    # pytest test suite — 16 test cases
├── requirements.txt    # All dependencies with versions
└── README.md           # Project documentation
```

**Why separated into app.py and debugger.py?**
Clean architecture principle — UI and business logic should never be in the same file. This makes the core logic independently testable without running the UI, which is why pytest can test `debugger.py` directly.

---

##  How FAISS Works in This Project

```python
# Step 1: Text errors are converted to 384-dimensional vectors
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 2: Vectors are stored in FAISS index
vectorstore = LC_FAISS.from_texts(error_texts, embedding_model)

# Step 3: When user pastes code, FAISS finds closest error by cosine similarity
docs = retriever.invoke(user_code)

# Step 4: Retrieved error is injected into LLM prompt as context (RAG)
prompt = f"Similar errors: {context}\nAnalyze: {user_code}\nGive: Error, Reason, Fix"
```

FAISS does **not** do keyword matching. It computes the cosine distance between the query vector and all stored vectors — finding the most semantically similar error even when exact words don't match.

---

##  Known Limitations & Planned Improvements

| Current Limitation | Planned Fix |
|---|---|
| `exec()` runs arbitrary code — security risk | Replace with Docker sandboxed execution |
| `distilgpt2` is a weak model | Swap with GPT-4 API or quantized LLaMA |
| Only 8 error patterns in FAISS | Expand to 500+ by scraping Stack Overflow |
| Python only | Add C++ and Java via subprocess compilation |
| No stack trace parsing | Parse full tracebacks, not just single exceptions |
| No feedback loop | Add thumbs up/down to learn from user corrections |

---

##  Relevance to QA Engineering

This project directly mirrors what QA Tools engineers do at companies like Nvidia:

| This Project | Nvidia QA Tools Team |
|---|---|
| Automates bug detection in Python code | Automates test failure detection in GPU drivers |
| RAG pipeline retrieves similar known errors | Test frameworks retrieve similar known failure patterns |
| 3-layer fallback ensures reliability | QA systems need guaranteed structured output |
| distilgpt2 generates fix suggestions | AI assists engineers in diagnosing test failures |
| pytest test suite validates the tool itself | QA tools must themselves be thoroughly tested |

---

##  Author

**Diya Mehta**
B.Tech CSE — MIT World Peace University, Pune
[GitHub](https://github.com/Diyamehta25)
