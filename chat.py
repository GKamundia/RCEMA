import os
import pandas as pd
import streamlit as st
import lancedb
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from transformers import TapasTokenizer, TapasForQuestionAnswering
import re
import numpy as np

# Load environment variables
load_dotenv()

# Retrieve your Hugging Face API token from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
# Set the generation model ID
GEN_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Initialize the Hugging Face endpoint for chat completions
llm = HuggingFaceEndpoint(
    repo_id=GEN_MODEL_ID,
    huggingfacehub_api_token=HF_TOKEN,
)

# Initialize database connection
@st.cache_resource
def init_db():
    """Initialize and return a LanceDB table object."""
    db = lancedb.connect(r"C:\Users\Anarchy\Documents\Data_Science\CEMA\RCEMA\docling\data\lancedb")
    return db.open_table("docling_tables")

# Set up TAPAS model for table-specific question answering
@st.cache_resource
def get_table_qa_model():
    tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
    model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")
    return model, tokenizer

# Helper function to convert markdown table to pandas DataFrame
def markdown_to_dataframe(markdown_table):
    """Convert a markdown table to pandas DataFrame"""
    lines = [line.strip() for line in markdown_table.split('\n') if line.strip().startswith('|')]
    if len(lines) < 3:
        return None
    
    # Extract headers
    headers = [h.strip() for h in lines[0].split('|')[1:-1]]
    
    # Create rows
    rows = []
    for line in lines[2:]:  # Skip separator line
        cells = [cell.strip() for cell in line.split('|')[1:-1]]
        if len(cells) == len(headers):
            rows.append(cells)
    
    # Create DataFrame
    return pd.DataFrame(rows, columns=headers)

# Function to check if a query is likely asking for numerical information
def is_numerical_query(query):
    """Check if the query is likely asking for numerical information"""
    patterns = [
        r'how many', r'number of', r'count of', r'total', r'sum of', 
        r'quantity', r'amount', r'figure', r'statistic', r'percentage',
        r'ratio', r'proportion'
    ]
    return any(re.search(pattern, query.lower()) for pattern in patterns)

# Function to analyze table with TAPAS
def answer_from_table(table_markdown, question):
    """Extract answer from table using TAPAS model"""
    model, tokenizer = get_table_qa_model()
    
    # Convert markdown to DataFrame
    df = markdown_to_dataframe(table_markdown)
    if df is None:
        return {"answer": None, "confidence": 0}
    
    # Handle number formatting - convert string numbers to numeric when possible
    for col in df.columns:
        # Try to convert to numeric, but keep original if not possible
        df[col] = pd.to_numeric(df[col].str.replace(',', '').replace('%', ''), errors='ignore')
    
    queries = [question]
    
    try:
        # Convert DataFrame to TAPAS input format with truncation to respect token limits
        inputs = tokenizer(
            table=df, 
            queries=queries, 
            padding="max_length",
            truncation=True,  # Add truncation to handle large tables
            max_length=512,   # Respect model's token limit
            return_tensors="pt"
        )
        outputs = model(**inputs)
        
        # Get predicted answer
        answers = tokenizer.convert_logits_to_answers(
            inputs=inputs,
            logits=outputs.logits.detach().numpy()
        )
        
        confidence = float(np.max(outputs.logits.softmax(dim=1).detach().numpy()))
        
        # For aggregate questions (totals, averages, etc.), enhance the answer with verification
        if 'total' in question.lower() or 'sum' in question.lower() or 'all' in question.lower():
            # Try to verify the answer by computing directly
            try:
                # For numeric columns that might be relevant to the question
                relevant_cols = [col for col in df.columns 
                               if any(term in question.lower() for term in [col.lower(), 'number', 'count', 'nurses', 'doctors'])]
                
                if relevant_cols:
                    # Try to sum the column if it contains numeric values
                    for col in relevant_cols:
                        if pd.to_numeric(df[col], errors='coerce').notna().all():
                            total = pd.to_numeric(df[col], errors='coerce').sum()
                            # If the computed total has higher confidence, use it
                            if abs(float(answers[0]) - total) > 5:  # If significant difference
                                answers[0] = str(int(total))  # Use computed total instead
            except Exception as e:
                # If computation fails, stick with TAPAS answer
                pass
        
        return {
            "answer": answers[0],
            "confidence": confidence
        }
    except Exception as e:
        print(f"TAPAS error: {e}")
        return {"answer": None, "confidence": 0}

def get_context(query: str, table, num_results: int = 3) -> dict:
    """
    Search the LanceDB table for relevant context.
    Returns dict with contexts and tables separately.
    """
    # Prioritize table-containing chunks for numerical queries
    if is_numerical_query(query):
        table_results = table.search(query).where("metadata.tables.has_table == true").limit(num_results).to_pandas()
        if len(table_results) > 0:
            num_results = max(1, num_results - len(table_results))
        else:
            num_results = num_results
        text_results = table.search(query).where("metadata.tables.has_table == false").limit(num_results).to_pandas()
        results = pd.concat([table_results, text_results]) if not text_results.empty else table_results
    else:
        # Standard search for non-numerical queries
        results = table.search(query).limit(num_results).to_pandas()
    
    contexts = []
    tables = []
    
    for _, row in results.iterrows():
        text = row["text"]
        meta = row["metadata"]
        
        # Format source information
        source = f"**Source:** {meta['filename']}"
        if meta["page_numbers"]:
            source += f" | **Pages:** {', '.join(map(str, meta['page_numbers']))}"
        if meta["title"]:
            source += f" | **Section:** {meta['title']}"
        
        # Is this primarily a table?
        is_table_chunk = meta["tables"]["has_table"] and meta["tables"]["table_count"] > 0
        
        if is_table_chunk:
            # Extract table content
            table_lines = [line for line in text.split('\n') if line.strip().startswith('|')]
            if len(table_lines) >= 3:  # Valid table needs header, separator and data rows
                table_text = '\n'.join(table_lines)
                table_info = {
                    "text": table_text,
                    "source": source,
                    "title": meta["tables"].get("title") or meta["title"] or "Table",
                    "columns": meta["tables"]["columns"]
                }
                tables.append(table_info)
        
        contexts.append(f"{text}\n\n{source}")
    
    return {
        "contexts": "\n\n---\n\n".join(contexts),
        "tables": tables
    }

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

def get_chat_response(messages, context_data: dict) -> str:
    """
    Get a response using both TAPAS for tables and LLM for text.
    """
    query = messages[-1]["content"]
    
    # Check if we have tables to analyze
    tables = context_data.get("tables", [])
    if tables and is_numerical_query(query):
        # Process each table with TAPAS
        best_answer = None
        best_confidence = 0
        best_table = None
        
        for table_data in tables:
            table_text = table_data["text"]
            answer_result = answer_from_table(table_text, query)
            
            if answer_result["answer"] and answer_result["confidence"] > best_confidence:
                best_answer = answer_result["answer"]
                best_confidence = answer_result["confidence"]
                best_table = table_data
        
        if best_answer and best_confidence > 0.5:  # Confidence threshold
            response = (
                f"**Answer:** {best_answer}\n\n"
                f"**Source:** {best_table['title']}\n\n"
                f"```markdown\n{best_table['text']}\n```"
            )
            return response
    
    # Fall back to text-based LLM
    system_prompt = (
        "You are an assistant called 'RCEMA' and you are here to help a company known as CEMA (Center for Epidemiological Modelling and Analysis) that answers questions based solely on the provided context. "
        "Use only the information from the context to answer questions. If you're unsure or the context "
        "doesn't contain the relevant information, say so.\n\n"
        "When presenting tables:"
        "1. Always preserve markdown table formatting\n"
        "2. Explain table contents clearly\n"
        "3. Reference source information\n\n"
        f"Context:\n{context_data['contexts']}\n"
    )
    
    # Prepend the system prompt as a system message
    messages_with_context = [{"role": "system", "content": system_prompt}] + messages
    
    # Combine messages into a single prompt string, ending with "Assistant:"
    combined_prompt = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages_with_context]
    ) + "\nAssistant:"
    
    # Call invoke with the combined prompt
    response = llm.invoke(
        input=combined_prompt, 
        temperature=0.7, 
        stop=["\nUser:", "```"]
    )
    
    return format_response(response)

# --------------------------------------------------------------
# Streamlit Chatbot UI
# --------------------------------------------------------------
st.title("📊 RCEMA - Table Q&A System")

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
div[data-testid="stMarkdownContainer"] code {
    white-space: pre-wrap !important;
}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

table = init_db()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if prompt := st.chat_input("Ask about document tables or content"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.status("Analyzing documents..."):
        context_data = get_context(prompt, table)
        
        if context_data["tables"]:
            st.write(f"Found {len(context_data['tables'])} relevant tables")
            for table_data in context_data["tables"]:
                with st.expander(f"{table_data['title']}"):
                    st.markdown(f"```markdown\n{table_data['text']}\n```")
    
    with st.chat_message("assistant"):
        response = get_chat_response(st.session_state.messages, context_data)
        st.markdown(response, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": response})
