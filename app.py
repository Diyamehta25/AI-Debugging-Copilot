import streamlit as st
from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Set Streamlit page config
st.set_page_config(page_title="AI Debugging Copilot", layout="centered")

# Load AI model
generator = pipeline("text-generation", model="distilgpt2")

# LangChain Embeddings + FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

texts = [
    "IndexError: list index out of range",
    "ZeroDivisionError: division by zero",
    "SyntaxError: unexpected EOF",
    "TypeError: unsupported operand type"
]

vectorstore = LC_FAISS.from_texts(texts, embedding_model)
retriever = vectorstore.as_retriever()

# Debugging function using LangChain + AI + Fallback
def debug_with_langchain(code):
    # Try executing the code first
    try:
        exec(code)
        return "No errors found in the code. The code runs successfully."
    except Exception as e:
        error_message = str(e)
        
        # Retrieve similar errors using FAISS
        docs = retriever.invoke(code)
        context = "\n".join([doc.page_content for doc in docs])

        # AI analysis
        try:
            prompt = f"""
You are an expert Python debugger.

Similar errors:
{context}

Now analyze the following code and explain the issue:
{code}

Give:
Error:
Reason:
Fix:
"""
            result = generator(prompt[:200], max_new_tokens=50, do_sample=False)
            output = result[0]['generated_text']

            # Clean output
            lines = output.split("\n")
            final = []

            for line in lines:
                if line.strip().startswith(("Error:", "Reason:", "Fix:")):
                    final.append(line.strip())
                if len(final) == 3:
                    break

            if len(final) == 3:
                return "\n".join(final)

        except:
            pass

        # Fallback logic for specific errors
        if "index" in error_message.lower():
            return """Error: IndexError: list index out of range
Reason: Trying to access an element from an empty or invalid index list
Fix: Check if the list is not empty before accessing elements"""

        if "division" in error_message.lower():
            return """Error: ZeroDivisionError: division by zero
Reason: Division by zero is not allowed
Fix: Ensure denominator is not zero before division"""

        if "syntax" in error_message.lower():
            return """Error: SyntaxError
Reason: Incorrect or incomplete syntax
Fix: Check for missing brackets or quotes"""

        if "type" in error_message.lower():
            return """Error: TypeError
Reason: Invalid operation between incompatible data types
Fix: Ensure operands are of compatible types"""

        return f"""Error: {error_message}
Reason: Based on similar bug patterns
Fix: Review your code logic"""

# Sidebar
st.sidebar.title("About")
st.sidebar.info("""
This AI Debugging Copilot uses:
- LangChain
- FAISS
- Sentence Embeddings

Paste your code and get instant fixes!
""")

# Header
st.title("AI Debugging Copilot")
st.markdown("Fix your code instantly with AI")

st.divider()

# Input area
code_input = st.text_area(
    "Paste your Python code below:",
    height=200,
    placeholder="e.g.\narr = []\nprint(arr[0])"
)

# Button and output
if st.button("Debug Code"):
    if code_input:
        with st.spinner("Analyzing your code..."):
            result = debug_with_langchain(code_input)

        st.success("Analysis Complete")
        st.markdown("Debugging Result")
        st.code(result, language="text")

    else:
        st.warning("Please enter some code to analyze.")

# Footer
st.divider()
st.markdown("Built with AI | Streamlit App")