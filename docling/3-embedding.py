from typing import List, Optional, Dict, Any, Union
import chromadb
from chromadb.utils import embedding_functions
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter, PdfFormatOption, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from dotenv import load_dotenv
from utils.tokenizer import HybridTokenizer
import re
import pandas as pd
import json
import os

# Load environment variables
load_dotenv()

# Initialize components
tokenizer = HybridTokenizer()
MAX_TOKENS = 512  # Adjust to match the model's limitations 

# Load your HF token from .env file
HF_TOKEN = os.getenv("HF_TOKEN")

# Enhanced table detection function
def extract_tables_from_markdown(text: str) -> List[Dict[str, Any]]:
    """Extract tables from markdown text with enhanced metadata"""
    # Pattern to find markdown tables (starts with | and has multiple rows)
    table_pattern = r'((?:\|[^\n]*\|\n)(?:\|[\s:-]+\|\n)(?:\|[^\n]*\|\n)+)'
    tables = []
    
    for match in re.finditer(table_pattern, text):
        table_text = match.group(1)
        
        # Extract table title (looks before the table for headings)
        title = "Table"
        title_match = re.search(r'#+\s+(.+?)(?:\n|$)', text[:match.start()].split('\n\n')[-1])
        if title_match:
            title = title_match.group(1).strip()
            
        # Extract column names
        lines = table_text.split('\n')
        if len(lines) >= 2 and lines[0].startswith('|'):
            columns = [col.strip() for col in lines[0].split('|')[1:-1]]
            
            # Try to convert to pandas for better analysis
            try:
                # Convert markdown table to pandas DataFrame for validation
                df = pd.read_csv(pd.StringIO(table_text), sep='|', skipinitialspace=True)
                df = df.iloc[:, 1:-1] if len(df.columns) > 2 else df
                
                tables.append({
                    "table_text": table_text,
                    "title": title,
                    "columns": columns,
                    "row_count": len(lines) - 2,  # Excluding header and separator
                    "has_numbers": any(col for col in df.columns 
                                      if pd.to_numeric(df[col], errors='coerce').notna().any())
                })
            except Exception:
                # Fallback if pandas conversion fails
                tables.append({
                    "table_text": table_text,
                    "title": title,
                    "columns": columns,
                    "row_count": len(lines) - 2
                })
    
    return tables

# Enhanced PDF processing
def create_converter():
    pipeline_options = PdfPipelineOptions(
        do_table_structure=True,
        table_structure_options={
            "mode": TableFormerMode.ACCURATE,
            "do_cell_matching": True
        }
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

# Process PDF files
pdf_files = ["C:/Users/Anarchy/Documents/Data_Science/CEMA/RCEMA/docling/Protocol on Alarm Fatigue May_26_2024.pdf"]

all_chunks = []
converter = create_converter()

for pdf_file in pdf_files:
    result = converter.convert(pdf_file)
    document = result.document
    
    # Verify table extraction
    markdown_output = document.export_to_markdown()
    print(f"Extracted content from {pdf_file}:\n{markdown_output[:500]}...")  # Show first 500 chars
    
    # Extract tables for special handling
    tables = extract_tables_from_markdown(markdown_output)
    print(f"Found {len(tables)} tables in document")
    
    # Save entire tables as separate chunks
    for table in tables:
        # Create a special table chunk with enhanced metadata
        all_chunks.append({
            "text": table["table_text"],
            "is_table": True,
            "table_meta": table,
            "page_numbers": [],  # Will be filled from metadata later
            "origin": pdf_file,
            "headings": [table["title"]]
        })
    
    # Chunk with table preservation for regular text
    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=MAX_TOKENS,
        merge_peers=False,  # Prevent merging across table boundaries
        split_long_sentences=False  # Keep table structures intact
    )
    
    text_chunks = list(chunker.chunk(document))
    
    # Find page numbers for tables by matching content
    for table in all_chunks:
        if table["is_table"]:
            table_text = table["text"]
            # Try to find which chunks contain this table
            for chunk in text_chunks:
                if table_text in chunk.text:
                    table["page_numbers"] = list(set(p.page_no for item in chunk.meta.doc_items for p in item.prov))
                    break
    
    # Filter out chunks that are just tables (to avoid duplication)
    filtered_chunks = []
    for chunk in text_chunks:
        has_full_table = False
        for table in tables:
            if table["table_text"] in chunk.text and len(chunk.text.strip()) - len(table["table_text"].strip()) < 200:
                has_full_table = True
                break
        if not has_full_table:
            filtered_chunks.append(chunk)
    
    # Process regular chunks
    for chunk in filtered_chunks:
        all_chunks.append({
            "text": chunk.text,
            "is_table": False,
            "table_meta": {},
            "page_numbers": list(set(p.page_no for item in chunk.meta.doc_items for p in item.prov)),
            "origin": chunk.meta.origin.filename,
            "headings": chunk.meta.headings
        })
    
    print(f"Added {len(filtered_chunks)} text chunks and {len(tables)} table chunks from {pdf_file}")

print(f"Total chunks processed: {len(all_chunks)}")

# Install required packages
# pip install sentence-transformers

# Replace the HuggingFace API embedding function with a local one
from sentence_transformers import SentenceTransformer
import numpy as np

# Create a local embedding function that doesn't require API calls
class LocalSentenceTransformerEmbedding:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
    def __call__(self, input):
        # Convert embeddings to list of lists
        return self.model.encode(input, show_progress_bar=True).tolist()

# Use the local embedding function instead of HuggingFace API
embedding_function = LocalSentenceTransformerEmbedding(model_name="all-MiniLM-L6-v2")

# Replace the ChromaDB collection creation code
chroma_client = chromadb.PersistentClient("data/chromadb")

# Delete collection if it exists (for clean restart)
try:
    chroma_client.delete_collection("docling_tables")
except:
    pass

# Create a new collection with local embedding function
collection = chroma_client.create_collection(
    name="docling_tables",
    embedding_function=embedding_function  # Local embedding function
)

# Prepare documents, metadatas and IDs for ChromaDB
documents = []
metadatas = []
ids = []

for i, chunk in enumerate(all_chunks):
    documents.append(chunk["text"])
    
    # Create nested tables structure for metadata
    tables_metadata = {
        "has_table": chunk["is_table"],
        "table_count": 1 if chunk["is_table"] else 0,
        "columns": chunk["table_meta"].get("columns", []) if chunk["is_table"] else [],
        "title": chunk["table_meta"].get("title", "") if chunk["is_table"] else "",
        "row_count": chunk["table_meta"].get("row_count", 0) if chunk["is_table"] else 0,
        "has_numbers": chunk["table_meta"].get("has_numbers", False) if chunk["is_table"] else False
    }
    
    # ChromaDB requires all metadata values to be strings
    metadata = {
        "filename": str(chunk["origin"]),
        "page_numbers": json.dumps(chunk["page_numbers"]),
        "title": str(chunk["headings"][0] if chunk["headings"] else ""),
        "tables": json.dumps(tables_metadata)
    }
    
    metadatas.append(metadata)
    ids.append(f"chunk_{i}")

# Process documents in smaller batches to avoid timeouts
BATCH_SIZE = 20  # Adjust this number as needed
for i in range(0, len(documents), BATCH_SIZE):
    batch_docs = documents[i:i+BATCH_SIZE]
    batch_metadata = metadatas[i:i+BATCH_SIZE]
    batch_ids = ids[i:i+BATCH_SIZE]
    
    print(f"Processing batch {i//BATCH_SIZE + 1}/{len(documents)//BATCH_SIZE + 1} ({len(batch_docs)} documents)")
    
    try:
        collection.add(
            documents=batch_docs,
            metadatas=batch_metadata,
            ids=batch_ids
        )
    except Exception as e:
        print(f"Error processing batch {i//BATCH_SIZE + 1}: {e}")
        # Continue with the next batch

print(f"Database populated with {collection.count()} entries")

# Test a simple search
results = collection.query(
    query_texts=["What is alarm fatigue?"],
    n_results=3
)

print("\nSample search results:")
for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
    print(f"\n--- Result {i+1} ---")
    print(f"Text: {doc[:150]}..." if len(doc) > 150 else doc)
    print(f"Metadata: {metadata}")
    print(f"Is table: {'Yes' if json.loads(metadata['tables'])['has_table'] else 'No'}")
