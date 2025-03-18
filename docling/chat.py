import os
import pandas as pd
import streamlit as st
import lancedb
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
import lancedb
import pandas as pd

def init_db():
    db = lancedb.connect(r"C:\Users\Anarchy\Documents\Data_Science\CEMA\RCEMA\docling\data\lancedb")
    return db.open_table("docling_tables")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Chat", "Database Explorer"])

if page == "Database Explorer":
    st.header("📊 Database Contents")
    table = init_db()
    df = table.to_pandas()
    
    # Process metadata columns
    df['filename'] = df.metadata.apply(lambda x: x['filename'])
    df['page_numbers'] = df.metadata.apply(lambda x: x['page_numbers'])
    df['has_table'] = df.metadata.apply(lambda x: x['tables']['has_table'])
    df['columns'] = df.metadata.apply(lambda x: x['tables']['columns'])
import lancedb
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables from .env
load_dotenv()

# Retrieve your Hugging Face API token from environment variables.
HF_TOKEN = os.getenv("HF_TOKEN")
# Set the generation model ID (this example uses a Mistralai model)
GEN_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Initialize the Hugging Face endpoint for chat completions.
llm = HuggingFaceEndpoint(
    repo_id=GEN_MODEL_ID,
    huggingfacehub_api_token=HF_TOKEN,
)

@st.cache_resource
def init_db():
    """
    Initialize and return a LanceDB table object.
    """
    db = lancedb.connect(r"C:\Users\Anarchy\Documents\Data_Science\CEMA\RCEMA\docling\data\lancedb")
    return db.open_table("docling_tables")

def get_context(query: str, table, num_results: int = 3) -> str:
    """
    Search the LanceDB table for relevant context.
    Returns concatenated text with source information.
    """
    # Prioritize table-containing chunks in search results
    results = table.search(query).limit(num_results).to_pandas()
    contexts = []
    for _, row in results.iterrows():
        text = row["text"]
        meta = row["metadata"]
        
        # Format source information
        source = f"**Source:** {meta['filename']}"
        if meta["page_numbers"]:
            source += f" | **Pages:** {', '.join(map(str, meta['page_numbers']))}"
        if meta["title"]:
            source += f" | **Section:** {meta['title']}"
        
        # Add table metadata
        if meta["tables"]["has_table"]:
            source += f" | **Contains:** {meta['tables']['table_count']} table(s)"
            source += f" | **Columns:** {', '.join(meta['tables']['columns'])}"
        
        contexts.append(f"{text}\n\n{source}")
    
    return "\n\n---\n\n".join(contexts)

def format_response(text: str) -> str:
    """Format tables in response"""
    in_table = False
    formatted = []
    for line in text.split('\n'):
        if line.startswith('|'):
            if not in_table:
                formatted.append("```markdown")
                in_table = True
            formatted.append(line)
        else:
            if in_table:
                formatted.append("```")
                in_table = False
            formatted.append(line)
    if in_table:
        formatted.append("```")
    return '\n'.join(formatted)

def get_chat_response(messages, context: str) -> str:
    """
    Get a chat completion response from the Hugging Face endpoint.
    This version constructs the prompt by prepending system instructions
    (which are not added to the displayed conversation history) to the
    conversation history.
    """
    system_prompt = (
        "You are an assistant called 'RCEMA' and you are here to help a company known as CEMA (Center for Epidemiological Modelling and Analysis) that answers questions based solely on the provided context. "
        "Use only the information from the context to answer questions. If you're unsure or the context "
        "doesn't contain the relevant information, say so.\n\n"
        "When presenting tables:"
        "1. Always preserve markdown table formatting\n"
        "2. Explain table contents clearly\n"
        "3. Reference source information\n\n"
        f"Context:\n{context}\n"
    )
    # Prepend the system prompt as a system message
    messages_with_context = [{"role": "system", "content": system_prompt}] + messages
    # Combine messages into a single prompt string, ending with "Assistant:"
    combined_prompt = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages_with_context]
    ) + "\nAssistant:"
    # Call invoke with the combined prompt.
    response = llm.invoke(
        input=combined_prompt, 
        temperature=0.7, 
        stop = ["\nUser:", "```"]) #Prevents table cuts
    return format_response(response)

# --------------------------------------------------------------
# Streamlit Chatbot UI
# --------------------------------------------------------------
st.title("📚 RCEMA")

# Add custom CSS for table styling
st.markdown("""
<style>
div[data-testid="stMarkdownContainer"] table {
    width: 100%;
    border-collapse: collapse;
    margin: 1em 0;
}
div[data-testid="stMarkdownContainer"] th {
    background-color: #f0f2f6;
    font-weight: 600;
}
div[data-testid="stMarkdownContainer"] td, th {
    padding: 8px;
    border: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

table = init_db()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if prompt := st.chat_input("Ask about document tables"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.status("Analyzing documents..."):
        context = get_context(prompt, table)
        st.write("Relevant tables found:")
        
        # Display raw table previews
        for chunk in context.split("\n\n---\n\n"):
            text_part = chunk.split('\n\n')[0]
            if '|' in text_part:
                st.markdown(f"**Extracted Table Preview:**\n```markdown\n{text_part}\n```")
    
    with st.chat_message("assistant"):
        response = get_chat_response(st.session_state.messages, context)
        st.markdown(response, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": response})
