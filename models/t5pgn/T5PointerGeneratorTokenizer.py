import torch
from transformers import AutoTokenizer
from typing import Optional, List, Tuple

class T5PointerGeneratorTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.unk_token_id = tokenizer.unk_token_id

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return cls(tokenizer)

    def decode_extended_ids(self, extended_ids: List[int], local_vocab: dict) -> str:
        tokens = []
        for extended_id in extended_ids:
            if extended_id < self.vocab_size:  # If it's a regular token ID
                tokens.append(self.tokenizer.convert_ids_to_tokens([extended_id])[0])
            else:  # If it's an extended token ID
                for token, token_id in local_vocab.items():
                    if token_id == extended_id:
                        tokens.append(token)
                        break
        return self.tokenizer.convert_tokens_to_string(tokens)

    def encode_extended_ids(
            self,
            text: str,
            max_length: Optional[int] = None,
            return_tensors: Optional[bool] = None,
            add_special_tokens: Optional[bool] = None,
        ) -> Tuple[List[int], dict]:

        encoding = self.tokenizer(
            text, 
            return_offsets_mapping=True, 
            add_special_tokens=add_special_tokens, 
            max_length=max_length,
            truncation=True)
        
        token_ids = encoding['input_ids'] # with unk id
        offsets = encoding['offset_mapping']

        # Create a local vacab mapping for OOV words
        local_vocab = {} 
        next_extended_id = self.vocab_size  # extended vocab IDs start from vocab_size
        for i, (start, end) in enumerate(offsets):
            if token_ids[i] == self.unk_token_id:  # If token is <unk>
                word = text[start:end]  # Get the original word
                if word not in local_vocab:
                    local_vocab[word] = next_extended_id
                    next_extended_id += 1
                token_ids[i] = local_vocab[word]  # Replace with extended ID

        if return_tensors:
            token_ids = torch.tensor(token_ids, dtype=torch.int32)

        return token_ids, local_vocab

    def encode_with_extended_vocab(
            self,
            text: str,
            local_vocab: dict,
            max_length: Optional[int] = None,
            return_tensors: Optional[bool] = None,
            add_special_tokens: Optional[bool] = None,
        ) -> Tuple[List[int], dict]:

        encoding = self.tokenizer(
            text, 
            return_offsets_mapping=True, 
            add_special_tokens=add_special_tokens, 
            max_length=max_length,
            truncation=True)
        
        token_ids = encoding['input_ids'] # with unk id
        offsets = encoding['offset_mapping']

        for i, (start, end) in enumerate(offsets):
            word = text[start:end]
            if token_ids[i] == self.unk_token_id and word in local_vocab:  # If token is <unk>
                token_ids[i] = local_vocab[word]  # Replace with extended ID

        if return_tensors:
            token_ids = torch.tensor(token_ids, dtype=torch.int32)

        return token_ids