from models.T5PointerGenerator import *
from transformers import GenerationConfig, T5Config
import torch

class T5PG4CarDataPreprocessor:
    def __init__(self, tokenizer: T5PointerGeneratorTokenizer, generation_config : GenerationConfig):
        self.tokenizer : T5PointerGeneratorTokenizer = tokenizer
        self.max_seq_len = generation_config.max_length
        self.max_decoder_seq_len = generation_config.max_new_tokens

    def __call__(self, data : dict):

        input_ids, local_vocab = self.tokenizer.encode_extended_ids(
            data['x'],
            max_length=self.max_seq_len,
            return_tensors=False,
            add_special_tokens=False,
        )

        labels = self.tokenizer.encode_with_extended_vocab(
            data['y'],
            max_length=self.max_decoder_seq_len,
            local_vocab=local_vocab,
            return_tensors=False,
            add_special_tokens=True,
        )

        return {
            'input_ids': input_ids, # with extended vocab ids (no unk id)
            'labels': labels, # with extended vocab ids (no unk id)
            'extended_vocab_size': len(local_vocab),
            'local_vocab': local_vocab,
            'x': data['x'],
            'y': data['y'],
        }


class T5PG4CarDataCollator:
    def __init__(self, config: T5Config):
        self.pad_token_id = config.pad_token_id
        self.padding_mode = 'right'

    def __call__(self, features):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(data['input_ids']) for data in features],
            batch_first=True,
            padding_value=self.pad_token_id,
            padding_side=self.padding_mode
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(data['labels']) for data in features],
            batch_first=True,
            padding_value=self.pad_token_id,
            padding_side=self.padding_mode
        )

        decoder_input_ids = labels[:, :-1]
        labels = labels[:, 1:]
        extended_vocab_size = max([data['extended_vocab_size'] for data in features])

        local_vocabs = None
        if 'local_vocab' in features[0]: # for evaluation
            local_vocabs = [data['local_vocab'] for data in features]

        return {
            'input_ids': input_ids,
            'attention_mask': (input_ids != self.pad_token_id).float(),
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': (decoder_input_ids != self.pad_token_id).float(),
            'labels': labels,
            'extended_vocab_size': extended_vocab_size,
            'local_vocabs': local_vocabs,
        }