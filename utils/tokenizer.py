from typing import Dict, List, Optional
from transformers import AutoTokenizer, TapasTokenizer

class HybridTokenizer:
    """Enhanced tokenizer with specialized table handling capabilities"""
    
    def __init__(self, 
                 text_model: str = "sentence-transformers/all-mpnet-base-v2", 
                 table_model: str = "google/tapas-base"):
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.table_tokenizer = TapasTokenizer.from_pretrained(table_model)
        self._vocab_size = len(self.text_tokenizer)
        # Set max length per model constraints
        self.max_length = 512  # Both models have 512 token limit

    def tokenize_text(self, text: str) -> list:
        """Tokenize regular text content"""
        return self.text_tokenizer.tokenize(text)

    def tokenize_table(self, table_markdown: str) -> dict:
        """Convert markdown table to TAPAS-compatible format and tokenize"""
        # Parse markdown table into a structure TAPAS can process
        try:
            # Simple conversion of markdown table to TAPAS input format
            rows = [row.strip() for row in table_markdown.split('\n') if row.strip().startswith('|')]
            if len(rows) < 3:  # Need header, separator, and at least one row
                return self.text_tokenizer(table_markdown, return_tensors="pt")
                
            # Extract headers and data
            headers = [h.strip() for h in rows[0].split('|')[1:-1]]
            data = []
            for row in rows[2:]:  # Skip separator row
                cells = [cell.strip() for cell in row.split('|')[1:-1]]
                if cells:
                    data.append(cells)
                    
            # Create TAPAS input
            return self.table_tokenizer(
                table=data,
                queries=[""],  # Empty query placeholder
                column_names=headers,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
        except Exception:
            # Fall back to text tokenization if table parsing fails
            return self.text_tokenizer(table_markdown, return_tensors="pt")

    def encode_text(self, text: str, **kwargs) -> list:
        """Encode text into token IDs"""
        return self.text_tokenizer.encode(text, **kwargs)

    def encode_table(self, table_markdown: str, **kwargs) -> dict:
        """Encode table into TAPAS-compatible format"""
        return self.tokenize_table(table_markdown)

    def get_combined_vocab(self) -> dict:
        """Merge vocabularies from both tokenizers"""
        return {**self.text_tokenizer.get_vocab(), **self.table_tokenizer.get_vocab()}

    @property
    def text_vocab_size(self) -> int:
        return self.text_tokenizer.vocab_size

    @property
    def table_vocab_size(self) -> int:
        return self.table_tokenizer.vocab_size
