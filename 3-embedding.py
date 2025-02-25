from typing import List
import lancedb
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from utils.tokenizer import HuggingFaceTokenizerWrapper

#Load environment variables from .env file
load_dotenv()

# No need to initialize an OpenAI client since we are using the Hugging Face tokenizer locally.
tokenizer = HuggingFaceTokenizerWrapper()  # Load our custom Hugging Face tokenizer
MAX_TOKENS = 256  # Maximum sequence length for the chosen model

# --------------------------------------------------------------
# Extract the data
# --------------------------------------------------------------
pdf_files = [
    "C:/Users/Anarchy/Documents/Data_Science/CEMA/RCEMA/docling/Protocol on Alarm Fatigue May_26_2024.pdf", # 145 chunks
    "C:/Users/Anarchy/Documents/Data_Science/CEMA/RCEMA/docling/Kenya DHS.pdf"  # Update this path
]

all_chunks = []
converter = DocumentConverter()
    
# --------------------------------------------------------------
# Apply hybrid chunking
# --------------------------------------------------------------
chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS,
    merge_peers=True,
    split_long_sentences=True,  # Ensure long sentences are split
)

for pdf_file in pdf_files:
    result = converter.convert(pdf_file)
    document = result.document
    
    # Apply hybrid chunking
    chunk_iter = chunker.chunk(dl_doc=document)
    chunks = list(chunk_iter)
    print(f"Number of chunks for {pdf_file}: {len(chunks)}")
    all_chunks.extend(chunks)

print(f"Total chunks: {len(all_chunks)}")


# --------------------------------------------------------------
# Create a LanceDB database and table
# --------------------------------------------------------------
db = lancedb.connect("data/lancedb")

# Get the Hugging Face embedding function instead of the OpenAI one
func = get_registry().get("huggingface").create(name="sentence-transformers/all-MiniLM-L6-v2")

# Define a simplified metadata schema
class ChunkMetadata(LanceModel):
    """
    Fields must be ordered alphabetically as required by Pydantic.
    """
    filename: str | None
    page_numbers: List[int] | None
    title: str | None

# Define the main schema for our chunks
class Chunks(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # type: ignore
    metadata: ChunkMetadata

table = db.create_table("docling", schema=Chunks, mode="overwrite")

# --------------------------------------------------------------
# Prepare the chunks for the table
# --------------------------------------------------------------
processed_chunks = [
    {
        "text": chunk.text,
        "metadata": {
            "filename": chunk.meta.origin.filename,
            "page_numbers": [
                page_no
                for page_no in sorted(
                    set(
                        prov.page_no
                        for item in chunk.meta.doc_items
                        for prov in item.prov
                    )
                )
            ] or None,
            "title": chunk.meta.headings[0] if chunk.meta.headings else None,
        },
    }
    for chunk in all_chunks
]

# --------------------------------------------------------------
# Add the chunks to the table (automatically embeds the text)
# --------------------------------------------------------------
table.add(processed_chunks)

# --------------------------------------------------------------
# Load the table and inspect results
# --------------------------------------------------------------
print(table.to_pandas())
print(table.count_rows())
