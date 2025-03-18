from typing import Dict, List, Tuple
from transformers import AutoTokenizer, PreTrainedTokenizerBase, TapasTokenizer


class HuggingFaceTokenizerWrapper(PreTrainedTokenizerBase):
    """
    Handles both text and table tokenization with dual modes
    """
    
    def __init__(self,
                 text_model: str = "sentence-transformers/all-mpnet-base-v2",
                 table_model: str = "google/tapas-base"):
        """
        Initialize the tokenizer.
        
        Args:
            model_name: The Hugging Face model to use for tokenization.
            max_length: Maximum sequence length.
        """
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.table_tokenizer = TapasTokenizer.from_pretrained(table_model)
        self.max_text_length = 512
        self.max_table_length = 512

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Tokenize the input text."""
        tokens = self.tokenizer.tokenize(text, **kwargs)
        return tokens

    def _tokenize(self, text: str) -> List[str]:
        return self.tokenize(text)

    def encode(self, text: str, **kwargs) -> List[int]:
        """
        Encode the input text into a list of token IDs.
        We set add_special_tokens=False so that chunking counts only the raw tokens.
        """
        return self.tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=self.model_max_length, **kwargs)

    def _convert_token_to_id(self, token: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.tokenizer.convert_ids_to_tokens(index)

    def get_vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def save_vocabulary(self, *args, **kwargs) -> Tuple[str]:
        return self.tokenizer.save_vocabulary(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Class method to match Hugging Face's interface."""
        return cls(*args, **kwargs)
