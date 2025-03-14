import torch
from transformers import AutoTokenizer
from typing import Optional, List, Tuple, Union

class T5PointerGeneratorTokenizer:
    def __init__(self, tokenizer : AutoTokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.unk_token_id = tokenizer.unk_token_id
        self.bos_token_id = tokenizer.cls_token_id # Start
        self.eos_token_id = tokenizer.sep_token_id # End
        self.pad_token_id = tokenizer.pad_token_id

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return cls(tokenizer)
    
    def batch_convert_extended_ids_to_tokens(self, extended_ids: torch.Tensor, local_vocabs: List[dict]) -> str:
        batch_tokens = []
        for i in range(extended_ids.shape[0]):
            batch_tokens.append(self.convert_extended_ids_to_tokens(extended_ids[i], local_vocabs[i]))
        return batch_tokens

    def convert_extended_ids_to_tokens(self, extended_ids: Union[List[int], torch.Tensor], local_vocab: dict) -> str:
        tokens = []
        for extended_id in extended_ids:
            if extended_id == self.eos_token_id or extended_id == self.pad_token_id:
                break
            if extended_id < self.vocab_size:  # If it's a regular token ID
                tokens.append(self.tokenizer._convert_id_to_token(extended_id))
            else:  # If it's an extended token ID
                for token, token_id in local_vocab.items():
                    if token_id == extended_id:
                        tokens.append(token)
                        break
        return tokens
    
    def decode_extended_ids(self, extended_ids: Union[List[int], torch.Tensor], local_vocab: dict) -> str:
        tokens = self.convert_extended_ids_to_tokens(extended_ids, local_vocab)
        return self.tokenizer.convert_tokens_to_string(tokens)
    
    def batch_decode_extended_ids(self, extended_ids: torch.Tensor, local_vocabs: List[dict]) -> List[str]:
        batch_tokens = self.batch_convert_extended_ids_to_tokens(extended_ids, local_vocabs)
        return [self.tokenizer.convert_tokens_to_string(tokens) for tokens in batch_tokens]

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