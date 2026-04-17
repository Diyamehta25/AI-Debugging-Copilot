# debugger.py
# Core debugging logic - separated from UI for clean architecture

from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# ── Load Models ──────────────────────────────────────────────
generator = pipeline("text-generation", model="distilgpt2")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ── Knowledge Base ───────────────────────────────────────────
texts = [
    "IndexError: list index out of range",
    "ZeroDivisionError: division by zero",
    "SyntaxError: unexpected EOF while parsing",
    "TypeError: unsupported operand type",
    "NameError: name is not defined",
    "AttributeError: object has no attribute",
    "KeyError: key not found in dictionary",
    "ValueError: invalid literal for int()",
]

vectorstore = LC_FAISS.from_texts(texts, embedding_model)
retriever = vectorstore.as_retriever()

# ── Core Debug Function ──────────────────────────────────────
def debug_with_langchain(code: str) -> str:
    """
    Analyzes Python code and returns structured debugging output.
    
    3-Layer Architecture:
    Layer 1: exec() - actually runs code, catches real exception
    Layer 2: RAG + LLM - retrieves similar errors, generates fix
    Layer 3: Rule-based fallback - guarantees structured response
    
    Args:
        code: Python code string to debug
    Returns:
        Structured string with Error, Reason, Fix
    """

    # ── Layer 1: Actually run the code ───────────────────────
    if not code or not code.strip():
        return "Error: Empty Input\nReason: No code was provided\nFix: Please paste your Python code"

    try:
        exec(code)
        return "No errors found in the code. The code runs successfully."
    except Exception as e:
        error_message = str(e)

    # ── Layer 2: RAG + LLM Analysis ──────────────────────────
    try:
        docs = retriever.invoke(code)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""You are an expert Python debugger.
Similar errors:
{context}
Analyze this code and explain the issue:
{code}
Give:
Error:
Reason:
Fix:
"""
        result = generator(prompt[:200], max_new_tokens=50, do_sample=False)
        output = result[0]['generated_text']

        lines = output.split("\n")
        final = []
        for line in lines:
            if line.strip().startswith(("Error:", "Reason:", "Fix:")):
                final.append(line.strip())
            if len(final) == 3:
                break

        if len(final) == 3:
            return "\n".join(final)

    except Exception:
        pass

    # ── Layer 3: Rule-Based Fallback ─────────────────────────
    error_lower = error_message.lower()

    if "index" in error_lower:
        return (
            "Error: IndexError: list index out of range\n"
            "Reason: Trying to access an element at an invalid index\n"
            "Fix: Check if the list is not empty before accessing elements"
        )
    if "division" in error_lower or "zero" in error_lower:
        return (
            "Error: ZeroDivisionError: division by zero\n"
            "Reason: You are dividing a number by zero\n"
            "Fix: Add a check to ensure denominator is not zero before dividing"
        )
    if "syntax" in error_lower:
        return (
            "Error: SyntaxError\n"
            "Reason: Your code has incorrect or incomplete syntax\n"
            "Fix: Check for missing colons, brackets, or quotes"
        )
    if "type" in error_lower:
        return (
            "Error: TypeError\n"
            "Reason: Operation between incompatible data types\n"
            "Fix: Ensure both operands are of compatible types before the operation"
        )
    if "name" in error_lower:
        return (
            "Error: NameError\n"
            "Reason: You are using a variable that has not been defined\n"
            "Fix: Define the variable before using it"
        )
    if "key" in error_lower:
        return (
            "Error: KeyError\n"
            "Reason: The key you are accessing does not exist in the dictionary\n"
            "Fix: Use dict.get(key) instead of dict[key] to avoid this error"
        )
    if "attribute" in error_lower:
        return (
            "Error: AttributeError\n"
            "Reason: You are calling a method or attribute that does not exist\n"
            "Fix: Check the object type and verify the attribute name is correct"
        )
    if "value" in error_lower:
        return (
            "Error: ValueError\n"
            "Reason: A function received an argument of correct type but invalid value\n"
            "Fix: Validate input values before passing them to functions"
        )

    return (
        f"Error: {error_message}\n"
        "Reason: An unexpected error occurred in your code\n"
        "Fix: Review your code logic and check the error message above"
    )