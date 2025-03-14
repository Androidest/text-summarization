from .T5PointerGeneratorOutputs import T5PointerGeneratorOutputs
from .T5PointerGeneratorTokenizer import T5PointerGeneratorTokenizer
import torch
from transformers import T5ForConditionalGeneration, T5Config, AutoTokenizer, GenerationConfig
from transformers.generation.utils import GenerateOutput
from typing import Optional, Tuple, Union

class DataPreprocessorForT5PointerGenerator:
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

class DataCollatorForT5PointerGenerator:
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

        return { 
            'input_ids': input_ids,
            'attention_mask': (input_ids != self.pad_token_id).float(),
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': (decoder_input_ids != self.pad_token_id).float(),
            'labels': labels,
            'extended_vocab_size': extended_vocab_size,
        }

class T5PointerGeneratorModel(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.p_gen_linear = torch.nn.Linear(config.d_model, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.vocab_size = config.vocab_size
        
        tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        self.unk_token_id = tokenizer.vocab[tokenizer.special_tokens_map['unk_token']]
        self.bos_token_id = tokenizer.vocab[tokenizer.special_tokens_map['cls_token']]
        self.eos_token_id = tokenizer.vocab[tokenizer.special_tokens_map['sep_token']]
        self.pad_token_id = tokenizer.vocab[tokenizer.special_tokens_map['pad_token']]

    def forward(
        self, 
        input_ids: Optional[torch.LongTensor] = None, # with extended vocab ids (no unk id) 
        attention_mask: Optional[torch.FloatTensor] = None, # padding mask
        decoder_input_ids: Optional[torch.LongTensor] = None,  # with extended vocab ids (no unk id) 
        decoder_attention_mask: Optional[torch.BoolTensor] = None, # padding mask
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        extended_vocab_size: Optional[int] = None, # extended vocab build from OOV words for each sequence
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], T5PointerGeneratorOutputs]:

        encoder_seq_len = input_ids.shape[-1]
        batch_size, decoder_seq_len = decoder_input_ids.shape

        # Replace extended vocab ids with unk id
        parent_input_ids = input_ids.masked_fill(input_ids >= self.vocab_size, self.unk_token_id)
        parent_decoder_input_ids = decoder_input_ids.masked_fill(decoder_input_ids >= self.vocab_size, self.unk_token_id)
        
        # Compute parent(T5) forward pass
        outputs = super().forward(
            input_ids=parent_input_ids, 
            attention_mask=attention_mask,
            decoder_input_ids=parent_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            labels=None,  # Don't compute loss in the parent forward pass
            output_attentions=True,  # Require cross-attention weights for p_gen computation
            output_hidden_states=True,
            return_dict=True
        )

        # Compute attention distribution
        # cross_attentions:(num_layers, (batch_size, num_heads, decoder_seq_len, encoder_seq_len))
        last_cross_attention = outputs.cross_attentions[-1]  # (batch_size, num_heads, decoder_seq_len, encoder_seq_len)
        attention_dist = last_cross_attention.mean(dim=1)  # (batch_size, decoder_seq_len/summarized text, encoder_seq_len/original text)
        assert attention_dist.shape == (batch_size, decoder_seq_len, encoder_seq_len)

        # Compute p_gen
        last_hidden_state = outputs.decoder_hidden_states[-1]  # (batch_size, decoder_seq_len, d_model)
        p_gen = self.sigmoid(self.p_gen_linear(last_hidden_state))  # (batch_size, decoder_seq_len, 1)
        assert p_gen.shape == (batch_size, decoder_seq_len, 1)

        # Compute extended vocab distribution
        logits = outputs.logits
        expanded_vocab_size = extended_vocab_size + self.vocab_size
        zeros = torch.zeros(batch_size, decoder_seq_len, extended_vocab_size, dtype=logits.dtype, device=logits.device)
        vocab_dist = logits.softmax(dim=-1)
        expanded_vocab_dist = torch.cat([vocab_dist, zeros], dim=-1) 
        assert expanded_vocab_dist.shape == (batch_size, decoder_seq_len, expanded_vocab_size)

        # Compute final distribution
        attention_dist = p_gen * attention_dist
        expanded_vocab_dist = (1 - p_gen) * expanded_vocab_dist
        indexes = input_ids.unsqueeze(1).expand(-1, decoder_seq_len, -1)
        assert indexes.shape == (batch_size, decoder_seq_len, encoder_seq_len)
        expanded_vocab_dist.scatter_add_(dim=-1, index=indexes, src=attention_dist)

        # Compute the loss if 'labels' is provided. 
        # this mechanism is required by transformers.Trainer/Seq2SeqTrainer
        loss = None
        if labels is not None:
            # the probability(vocab distribution) is already calculated above.
            # hence, we only need to calculate the Negative Log-Likelihood Loss.
            # cause NLLLoss(vocab_dist, target) is equivalent to cross_entropy(logits, target)

            # Compute the Negative Log-Likelihood Loss
            selected_indexes = labels.unsqueeze(-1)
            selected_probs = expanded_vocab_dist.gather(dim=-1, index=selected_indexes).squeeze(-1)
            assert selected_probs.shape == (batch_size, decoder_seq_len)
            assert decoder_attention_mask.shape == (batch_size, decoder_seq_len)
            loss = -torch.log(selected_probs + 1e-10) * decoder_attention_mask
            loss = loss.sum() / decoder_attention_mask.sum() # average loss over non-padding tokens

        encoder_outputs = (outputs.encoder_last_hidden_state, outputs.encoder_hidden_states, outputs.encoder_attentions)
        # return dict
        return T5PointerGeneratorOutputs(
            loss=loss, # loss is required by transformers.Trainer/Seq2SeqTrainer
            logits=logits,
            expanded_vocab_dist=expanded_vocab_dist,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_outputs=encoder_outputs
        )

    @torch.no_grad()
    def generate(
        self, 
        input_ids: Optional[torch.LongTensor] = None, # with extended vocab ids (no unk id) 
        attention_mask: Optional[torch.FloatTensor] = None, # padding mask
        extended_vocab_size: Optional[int] = None, # extended vocab build from OOV words for each sequence
        labels: Optional[torch.LongTensor] = None,
        synced_gpus: Optional[bool] = False,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        
        self.eval()
        batch_size = input_ids.shape[0]
        gen_len = self.generation_config.max_new_tokens
        gen_len = self.generation_config.max_new_tokens
        batch_size = input_ids.shape[0]
        decoder_input_ids = torch.full((batch_size, 1), fill_value=self.bos_token_id, dtype=input_ids.dtype, device=input_ids.device)
        encoder_outputs = None
        finish_flags = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        for i in range(1, gen_len):
            decoder_attention_mask = (decoder_input_ids != self.pad_token_id).float()
            outputs = self.forward(
                input_ids=input_ids, # only used at the first step, when encoder_outputs is None
                attention_mask=attention_mask, # only used at the first step, when encoder_outputs is None
                decoder_input_ids=decoder_input_ids, 
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs, # reuse cached encoder_outputs, prevent recomputation of encoder_outputs for each step
                extended_vocab_size=extended_vocab_size, # oov token count
                labels=None, # Don't compute loss
            )

            # cache encoder_outputs
            encoder_outputs = outputs.encoder_outputs

            # Compute next token id
            scores = outputs.expanded_vocab_dist[:, -1:, :] # last token scores(extended vocab distribution)
            next_token_ids = torch.argmax(scores, dim=-1) # greedy search
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_ids], dim=-1)

            # stops generating when all sequences have eos token
            finish_flags |= (next_token_ids.squeeze(-1) == self.eos_token_id)
            if finish_flags.all():
                break

        return decoder_input_ids[:, 1:]
