from models.BertPointerGenerator import *
import torch

class BertPGN4CarDataPreprocessor:
    def __init__(self, tokenizer: BertPointerGeneratorTokenizer, config : BertPointerGeneratorConfig):
        self.tokenizer = tokenizer
        self.max_seq_len = config.max_length

    def __call__(self, data : dict):

        input_ids, local_vocab = self.tokenizer.encode_extended_ids(
            data['x'],
            max_length=self.max_seq_len,
            return_tensors=False,
            add_special_tokens=True,
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
        }


class BertPGN4CarDataCollator:
    def __init__(self, config: BertPointerGeneratorConfig, return_local_vocab=False):
        self.pad_token_id = config.pad_token_id
        self.padding_mode = 'right'
        self.return_local_vocab = return_local_vocab

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

        extended_vocab_size = max([data['extended_vocab_size'] for data in features])

        data = {
            'input_ids': input_ids,
            'attention_mask': (input_ids != self.pad_token_id).float(),
            'labels': labels,
            'extended_vocab_size': extended_vocab_size,
        }

        if self.return_local_vocab:
            data["local_vocabs"] = [data['local_vocab'] for data in features]

        return data