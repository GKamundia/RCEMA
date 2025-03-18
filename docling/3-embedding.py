from typing import List, Optional
import lancedb
import pydantic
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter,PdfFormatOption, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
from dotenv import load_dotenv
import pyarrow as pa
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from utils.tokenizer import HuggingFaceTokenizerWrapper

# Load environment variables
load_dotenv()

# Initialize components with table support
tokenizer = HuggingFaceTokenizerWrapper()
MAX_TOKENS = 512  # Increased to preserve table structures

# Enhanced PDF processing configuration
def create_converter():
    """Create document converter with table extraction settings"""
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

# --------------------------------------------------------------
# Process PDF files
# --------------------------------------------------------------
pdf_files = [
    "C:/Users/Anarchy/Documents/Data_Science/CEMA/RCEMA/docling/Protocol on Alarm Fatigue May_26_2024.pdf",
]

all_chunks = []
converter = create_converter()

for pdf_file in pdf_files:
    result = converter.convert(pdf_file)
    document = result.document
    
    # Verify table extraction
    markdown_output = document.export_to_markdown()
    print(f"Extracted content from {pdf_file}:\n{markdown_output[:500]}...")  # Show first 500 chars
    
    # Chunk with table preservation
    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=MAX_TOKENS,
        merge_peers=False,  # Prevent merging across table boundaries
        split_long_sentences=False  # Keep table structures intact
    )
    
    chunks = list(chunker.chunk(document))
    all_chunks.extend(chunks)
    print(f"Added {len(chunks)} chunks from {pdf_file}")

print(f"Total chunks processed: {len(all_chunks)}")

# --------------------------------------------------------------
# LanceDB Setup with Table Metadata and BERT Embeddings
# --------------------------------------------------------------
db = lancedb.connect("data/lancedb")
# Get embedding function and dimensions
embedding_func = get_registry().get("huggingface").create(name="bert-base-uncased") 
vector_dim = embedding_func.ndims()  # Get dimensions as integer

class TableMetadata(pydantic.BaseModel):
    has_table: bool
    table_count: int
    columns: List[str] = pydantic.Field(default_factory=list)

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

# --------------------------------------------------------------
# Process and Store Chunks
# --------------------------------------------------------------
processed_chunks = []
for chunk in all_chunks:
    # Detect tables in chunk text
    table_lines = [line for line in chunk.text.split('\n') if line.startswith('|')]
    has_table = len(table_lines) > 2  # At least header separator and one row
    
    metadata = {
        "filename": chunk.meta.origin.filename,
        "page_numbers": list(set(p.page_no for item in chunk.meta.doc_items for p in item.prov)),
        "title": chunk.meta.headings[0] if chunk.meta.headings else None,
        "tables": {
            "has_table": has_table,
            "table_count": chunk.text.count('\n|') // 3,  # Approximate table count
            "columns": table_lines[0].split('|')[1:-1] if has_table else []
        }
    }
    
    processed_chunks.append({
        "text": chunk.text,
        "metadata": metadata
    })

table.add(processed_chunks)
print(f"Database populated with {table.count_rows()} entries")
