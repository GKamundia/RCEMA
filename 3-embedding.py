from typing import List, Optional, Dict, Any
import lancedb
import pydantic
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter, PdfFormatOption, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from dotenv import load_dotenv
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from utils.tokenizer import HybridTokenizer
import re
import pandas as pd

# Load environment variables
load_dotenv()

# Initialize components
tokenizer = HybridTokenizer()
MAX_TOKENS = 1024  # Increased for table preservation

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

# LanceDB Setup with enhanced embedding model
db = lancedb.connect("data/lancedb")
embedding_func = get_registry().get("huggingface").create(
    name="sentence-transformers/all-mpnet-base-v2"  # Better semantic understanding than BERT
)
vector_dim = embedding_func.ndims()

class TableMetadata(pydantic.BaseModel):
    has_table: bool
    table_count: int
    columns: List[str] = pydantic.Field(default_factory=list)
    title: str = ""
    row_count: int = 0
    has_numbers: bool = False

class ChunkMetadata(LanceModel):
    filename: str
    page_numbers: Optional[List[int]] = pydantic.Field(default_factory=list)
    title: Optional[str] = None
    tables: TableMetadata

class Chunks(LanceModel):
    text: str = embedding_func.SourceField()
    vector: Vector(vector_dim) = embedding_func.VectorField() # type: ignore
    metadata: ChunkMetadata

table = db.create_table("docling_tables", schema=Chunks, mode="overwrite")

# Process and Store Chunks with enhanced metadata
processed_chunks = []
for chunk in all_chunks:
    if chunk["is_table"]:
        # Special handling for table chunks
        metadata = {
            "filename": chunk["origin"],
            "page_numbers": chunk["page_numbers"],
            "title": chunk["table_meta"].get("title", "Table"),
            "tables": {
                "has_table": True,
                "table_count": 1,
                "columns": chunk["table_meta"].get("columns", []),
                "title": chunk["table_meta"].get("title", "Table"),
                "row_count": chunk["table_meta"].get("row_count", 0),
                "has_numbers": chunk["table_meta"].get("has_numbers", False)
            }
        }
    else:
        # Regular text chunks
        table_lines = [line for line in chunk["text"].split('\n') if line.strip().startswith('|')]
        has_table = len(table_lines) > 2
        
        metadata = {
            "filename": chunk["origin"],
            "page_numbers": chunk["page_numbers"],
            "title": chunk["headings"][0] if chunk["headings"] else None,
            "tables": {
                "has_table": has_table,
                "table_count": chunk["text"].count('\n|') // 3,
                "columns": table_lines[0].split('|')[1:-1] if has_table else [],
                "title": "",
                "row_count": len(table_lines) - 2 if has_table else 0,
                "has_numbers": False
            }
        }
    
    processed_chunks.append({
        "text": chunk["text"],
        "metadata": metadata
    })

table.add(processed_chunks)
print(f"Database populated with {table.count_rows()} entries")
