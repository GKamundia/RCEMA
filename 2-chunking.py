from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from utils.tokenizer import HuggingFaceTokenizerWrapper

load_dotenv()

# No need to initialize an OpenAI client since we are using the Hugging Face tokenizer locally.
tokenizer = HuggingFaceTokenizerWrapper()  # Load our custom Hugging Face tokenizer
MAX_TOKENS = 256  # Maximum sequence length for the chosen model

# --------------------------------------------------------------
# Extract the data
# --------------------------------------------------------------
converter = DocumentConverter()
# Replace with the correct path to your PDF or document.
result = converter.convert("C:/Users/Anarchy/Documents/Data_Science/CEMA/RCEMA/docling/Protocol on Alarm Fatigue May_26_2024.pdf")

document = result.document
markdown_output = document.export_to_markdown()
print(markdown_output)


# --------------------------------------------------------------
# Apply hybrid chunking
# --------------------------------------------------------------
chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS,
    merge_peers=True,
    split_long_sentences=True,  # Ensure long sentences are split
)

chunk_iter = chunker.chunk(dl_doc=result.document)
chunks = list(chunk_iter)
print(f"Number of chunks: {len(chunks)}")
