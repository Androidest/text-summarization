from .T5PointerGeneratorOutputs import T5PointerGeneratorOutputs
from .T5PointerGeneratorTokenizer import T5PointerGeneratorTokenizer
from transformers import T5ForConditionalGeneration, T5Config, AutoTokenizer
from transformers.generation.utils import GenerateOutput
from typing import Optional, Tuple, Union
import torch

class T5PointerGeneratorModel(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.p_gen_linear = torch.nn.Linear(config.d_model, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
        tokenizer = T5PointerGeneratorTokenizer.from_pretrained(config._name_or_path)
        self.vocab_size = tokenizer.vocab_size
        self.unk_token_id = tokenizer.unk_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def forward(
        self, 
        input_ids: Optional[torch.LongTensor] = None, # with extended vocab ids (no unk id) 
        attention_mask: Optional[torch.FloatTensor] = None, # padding mask
        decoder_input_ids: Optional[torch.LongTensor] = None, # with extended vocab ids (no unk id) 
        decoder_attention_mask: Optional[torch.BoolTensor] = None, # padding mask
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        extended_vocab_size: Optional[int] = None, # extended vocab build from OOV words for each sequence
        copy_weight : float = 1, # weight of copy, for inference only: final_p_gen = p_gen * copy_weight
        output_attentions : Optional[bool] = None,  # Require cross-attention weights for p_gen computation
        output_hidden_states : Optional[bool] = None,
        return_dict : Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple[torch.FloatTensor], T5PointerGeneratorOutputs]:

        if input_ids is None:
            input_ids = self._input_ids
            
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
            return_dict=True,
            **kwargs
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
        final_p_gen = p_gen * copy_weight
        attention_dist = final_p_gen * attention_dist
        expanded_vocab_dist = (1 - final_p_gen) * expanded_vocab_dist
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
            logits=expanded_vocab_dist,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_outputs=encoder_outputs
        )
    
    @torch.no_grad()
    def generate__(
        self, 
        inputs: Optional[torch.LongTensor] = None, # with extended vocab ids (no unk id)
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
            Why override this function?:
                Because self.encoder(input_ids) would be called inside the generate() function
                the input_ids would be consumed by the encoder. 
                then the input_ids would be None when calling the self.forward() function at the next step.
                we need to cache the input_ids for the forward() function before calling the super().generate()
        """
        # cache input_ids for the forward() function
        self._input_ids = inputs
        # Replace extended vocab ids with unk id, 
        # because the extended vocab ids are illegal inputs for the encoder
        inputs = inputs.masked_fill(inputs >= self.vocab_size, self.unk_token_id)
        outputs = super().generate(inputs, **kwargs)
        return outputs

    @torch.no_grad()
    def generate(
        self, 
        inputs: Optional[torch.LongTensor] = None, # with extended vocab ids (no unk id) 
        attention_mask: Optional[torch.FloatTensor] = None, # padding mask
        extended_vocab_size: Optional[int] = None, # extended vocab build from OOV words for each sequence
        copy_weight: Optional[float] = 1, # weight of copy, for inference only: final_p_gen = p_gen * copy_weight
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        
        if self.training:
            self.eval()
        batch_size = inputs.shape[0]
        gen_len = self.generation_config.max_new_tokens
        decoder_input_ids = torch.full((batch_size, 1), fill_value=self.bos_token_id, dtype=inputs.dtype, device=inputs.device)
        encoder_outputs = None
        finish_flags = torch.zeros(batch_size, dtype=torch.bool, device=inputs.device)

        if attention_mask is None:
            attention_mask = (inputs!= self.pad_token_id).float()

        for i in range(1, gen_len):
            decoder_attention_mask = (decoder_input_ids != self.pad_token_id).float()
            outputs = self.forward(
                input_ids=inputs, # only used at the first step, when encoder_outputs is None
                attention_mask=attention_mask, # only used at the first step, when encoder_outputs is None
                decoder_input_ids=decoder_input_ids, 
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs, # reuse cached encoder_outputs, prevent recomputation of encoder_outputs for each step
                extended_vocab_size=extended_vocab_size, # oov token count for PGN
                labels=None, # Don't compute loss
                copy_weight=copy_weight,
                **kwargs
            )
            # cache encoder_outputs
            encoder_outputs = outputs.encoder_outputs

            # Compute next token id
            scores = outputs.logits[:, -1:, :] # PGN extented vocab distribution 0~1
            scores = self.repetition_penalty(scores, inputs.unsqueeze(1))

            next_token_ids = scores.argmax(dim=-1) # greedy search
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_ids], dim=-1)

            # stops generating when all sequences have eos token
            next_token_ids.squeeze_(-1)
            finish_flags |= (next_token_ids == self.eos_token_id) | (next_token_ids == self.pad_token_id)
            if finish_flags.all():
                break
            
        return decoder_input_ids[:, 1:]
    
    def repetition_penalty(self, scores: torch.Tensor, inputs: torch.Tensor) -> torch.FloatTensor:
        penalty = 20
        score = torch.gather(scores, -1, inputs)
        score = score / penalty
        scores_processed = scores.scatter(-1, inputs, score)
        return scores_processed