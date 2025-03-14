from .T5PointerGeneratorModel import T5PointerGeneratorModel
from .T5PointerGeneratorTokenizer import T5PointerGeneratorTokenizer
from .Arguments import RougeMetric
from transformers import GenerationConfig
from typing import Optional, Tuple, Union
import torch

class T5PointerGeneratorSummizer:
    def __init__(self, path : str):
        self.model = T5PointerGeneratorModel.from_pretrained(path)
        self.tokenizer = T5PointerGeneratorTokenizer.from_pretrained(path)
        self.generation_config = GenerationConfig.from_pretrained(path)
        self.model.eval()
        self.rouge_metrics = RougeMetric(self.tokenizer)

    def summarize(self, input_text : str) -> str:
        input_ids, local_vocab = self.tokenizer.encode_extended_ids(
            input_text,
            max_length=self.generation_config.max_seq_len,
            return_tensors=True,
            add_special_tokens=False)

        output_ids = self.model.generate(input_ids, extended_vocab_size=len(local_vocab))
        summirized_text = self.tokenizer.decode_extended_ids(output_ids, local_vocab)
        return summirized_text
    
    def evaluate(self, 
        input_ids: Optional[torch.LongTensor], # with extended vocab ids (no unk id) 
        attention_mask: Optional[torch.FloatTensor] = None, # padding mask
        extended_vocab_size: Optional[int] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> dict:
        
        predictions = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            extended_vocab_size=extended_vocab_size)
        
        scores = self.rouge_metrics((predictions, labels), compute_result=False)
        return scores
        
