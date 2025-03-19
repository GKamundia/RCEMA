from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, TapasTokenizer, PreTrainedTokenizerBase

class HybridTokenizer(PreTrainedTokenizerBase):
    """
    Hybrid tokenizer that handles both text and tables.
    Implements all required methods for compatibility with docling.
    """
    
    def __init__(self, 
                 text_model: str = "sentence-transformers/all-mpnet-base-v2", 
                 table_model: str = "google/tapas-base",
                 max_length: int = 512):
        """
        Initialize the tokenizer.
        
        Args:
            text_model: HF model for text tokenization
            table_model: HF model for table tokenization
            max_length: Maximum sequence length
        """
        # Initialize parent with required parameters
        super().__init__(
            model_max_length=max_length,
            padding_side="right",
            truncation_side="right",
        )
        
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.table_tokenizer = TapasTokenizer.from_pretrained(table_model)
        self._vocab_size = len(self.text_tokenizer)
        self.model_max_length = max_length
    
    # ---------- Core methods required by HF and docling ----------
    
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Public tokenize method that docling calls"""
        tokens = self.text_tokenizer.tokenize(text, **kwargs)
        # Handle truncation internally to avoid errors
        max_len = kwargs.get("max_length") or self.model_max_length
        if max_len and len(tokens) > max_len:
            tokens = tokens[:max_len]
        return tokens
    
    def _tokenize(self, text: str) -> List[str]:
        """Private tokenize method required by HF"""
        return self.text_tokenizer.tokenize(text)
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """
        Encode text to token IDs - critical for semantic chunking
        """
        # Default to False for chunking compatibility
        add_special = kwargs.pop("add_special_tokens", False)
        return self.text_tokenizer.encode(
            text, 
            add_special_tokens=add_special,
            truncation=True,
            max_length=self.model_max_length,
            **kwargs
        )
    
    def encode_plus(self, *args, **kwargs):
        """Required by semchunk"""
        kwargs["truncation"] = True
        kwargs["max_length"] = self.model_max_length
        return self.text_tokenizer.encode_plus(*args, **kwargs)
    
    def _encode_plus(self, *args, **kwargs):
        """Private method required by HF"""
        return self.text_tokenizer._encode_plus(*args, **kwargs)
    
    def __len__(self):
        return len(self.text_tokenizer)
    
    def _convert_token_to_id(self, token: str) -> int:
        return self.text_tokenizer.convert_tokens_to_ids(token)
    
    def _convert_id_to_token(self, index: int) -> str:
        return self.text_tokenizer.convert_ids_to_tokens(index)
    
    def get_vocab(self) -> Dict[str, int]:
        return self.text_tokenizer.get_vocab()
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return self.text_tokenizer.save_vocabulary(save_directory, filename_prefix)
    
    # ---------- Table-specific methods ----------
    
    def tokenize_text(self, text: str) -> list:
        """Tokenize regular text content"""
        return self.text_tokenizer.tokenize(text)
    
    def tokenize_table(self, table_markdown: str) -> dict:
        """Convert markdown table to TAPAS-compatible format and tokenize"""
        try:
            # Extract table structure from markdown
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
                max_length=self.model_max_length,
                return_tensors="pt"
            )
        except Exception:
            # Fall back to text tokenization if table parsing fails
            return self.text_tokenizer(table_markdown, return_tensors="pt")

    def encode_table(self, table_markdown: str, **kwargs) -> dict:
        """Encode table into TAPAS-compatible format"""
        return self.tokenize_table(table_markdown)
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Class method to match Hugging Face's interface."""
        return cls(*args, **kwargs)
