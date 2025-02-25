import os
import streamlit as st
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
    db = lancedb.connect("data/lancedb")
    return db.open_table("docling")

def get_context(query: str, table, num_results: int = 3) -> str:
    """
    Search the LanceDB table for relevant context.
    Returns concatenated text with source information.
    """
    results = table.search(query).limit(num_results).to_pandas()
    contexts = []
    for _, row in results.iterrows():
        filename = row["metadata"]["filename"]
        page_numbers = row["metadata"]["page_numbers"]
        title = row["metadata"]["title"]
        source_parts = []
        if filename:
            source_parts.append(filename)
        if page_numbers is not None:
            page_numbers_list = list(page_numbers)
            if len(page_numbers_list) > 0:
                source_parts.append(f"p. {', '.join(str(p) for p in page_numbers_list)}")
        source = f"\nSource: {' - '.join(source_parts)}"
        if title:
            source += f"\nTitle: {title}"
        contexts.append(f"{row['text']}{source}")
    return "\n\n".join(contexts)

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
        f"Context:\n{context}\n"
    )
    # Prepend the system prompt as a system message
    messages_with_context = [{"role": "system", "content": system_prompt}] + messages
    # Combine messages into a single prompt string, ending with "Assistant:"
    combined_prompt = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages_with_context]
    ) + "\nAssistant:"
    # Call invoke with the combined prompt.
    response = llm.invoke(input=combined_prompt, temperature=0.7, stop = ["\nUser:", "\nAssistant:"])
    return response.strip()

# --------------------------------------------------------------
# Streamlit Chatbot UI
# --------------------------------------------------------------
st.title("📚 RCEMA")

# Initialize chat history in session state.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the database connection.
table = init_db()

# Display existing chat messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input area.
if prompt := st.chat_input("Ask a question about the document"):
    # Display user's message.
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Retrieve relevant context from the document.
    with st.status("Searching document...", expanded=False):
        context = get_context(prompt, table)
        st.markdown(
            """
            <style>
            .search-result {
                margin: 10px 0;
                padding: 10px;
                border-radius: 4px;
                background-color: #f0f2f6;
            }
            .search-result summary {
                cursor: pointer;
                color: #0f52ba;
                font-weight: 500;
            }
            .search-result summary:hover {
                color: #1e90ff;
            }
            .metadata {
                font-size: 0.9em;
                color: #666;
                font-style: italic;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.write("Found relevant sections:")
        for chunk in context.split("\n\n"):
            parts = chunk.split("\n")
            text = parts[0]
            metadata = {
                line.split(": ")[0]: line.split(": ")[1]
                for line in parts[1:] if ": " in line
            }
            source = metadata.get("Source", "Unknown source")
            title = metadata.get("Title", "Untitled section")
            st.markdown(
                f"""
                <div class="search-result">
                    <details>
                        <summary>{source}</summary>
                        <div class="metadata">Section: {title}</div>
                        <div style="margin-top: 8px;">{text}</div>
                    </details>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    # Get and display the assistant's response.
    with st.chat_message("assistant"):
        response = get_chat_response(st.session_state.messages, context)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
