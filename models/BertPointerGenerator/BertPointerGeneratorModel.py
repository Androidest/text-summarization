from .BertPointerGeneratorOutputs import *
from .BertPointerGeneratorTokenizer import *
from transformers import BertModel, BertConfig
from transformers.generation.utils import GenerateOutput
from typing import Optional, Tuple, Union
import torch

class BertPointerGeneratorModel(BertModel):
    def __init__(self, config: BertConfig):
        super().__init__(config, add_pooling_layer=False)
        self.lm_head = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.p_gen_linear = torch.nn.Linear(config.d_model, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
        tokenizer = BertPointerGeneratorTokenizer.from_pretrained(config._name_or_path)
        self.vocab_size = tokenizer.vocab_size
        self.unk_token_id = tokenizer.unk_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def forward(
        self, 
        input_ids: Optional[torch.LongTensor] = None, # with extended vocab ids (no unk id) 
        attention_mask: Optional[torch.FloatTensor] = None, # padding mask
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        extended_vocab_size: Optional[int] = None, # extended vocab build from OOV words for each sequence
        output_attentions : Optional[bool] = None,  # Require cross-attention weights for p_gen computation
        output_hidden_states : Optional[bool] = None,
        return_dict : Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], BertPointerGeneratorOutputs]:

        if input_ids is None:
            input_ids = self._input_ids

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).float()
            
        batch_size, seq_len = input_ids.shape
        # Replace extended vocab ids with unk id
        parent_input_ids = input_ids.masked_fill(input_ids >= self.vocab_size, self.unk_token_id)

        outputs = super().forward(
            input_ids=parent_input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=True,  # Require cross-attention weights for p_gen computation
            output_hidden_states=True,
            return_dict=True,
            cache_position=cache_position,
        )

        # Compute attention distribution
        last_self_attention = outputs.attentions[-1] # (batch_size, num_heads, seq_len, encoder_seq_len)
        attention_dist = last_self_attention.mean(dim=1)  # (batch_size, seq_len/summarized text, encoder_seq_len/original text)

        # Compute p_gen
        last_hidden_state = outputs.last_hidden_state # (batch_size, seq_len, d_model)
        p_gen = self.sigmoid(self.p_gen_linear(last_hidden_state))  # (batch_size, seq_len, 1)

        # Compute extended vocab distribution
        logits = self.lm_head(last_hidden_state) # (batch_size, seq_len, vocab_size)
        vocab_dist = logits.softmax(dim=-1)
        zeros = torch.zeros(batch_size, seq_len, extended_vocab_size, dtype=logits.dtype, device=logits.device)
        expanded_vocab_dist = torch.cat([vocab_dist, zeros], dim=-1) 

        # Compute final distribution
        attention_dist = p_gen * attention_dist
        expanded_vocab_dist = (1 - p_gen) * expanded_vocab_dist
        indexes = input_ids.unsqueeze(1).expand(-1, seq_len, -1)
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
            loss = -torch.log(selected_probs + 1e-10) * attention_mask
            loss = loss.sum() / attention_mask.sum() # average loss over non-padding tokens

        # return dict
        return BertPointerGeneratorOutputs(
            loss=loss, # loss is required by transformers.Trainer/Seq2SeqTrainer
            logits=expanded_vocab_dist,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            past_key_values=outputs.past_key_values,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    @torch.no_grad()
    def generate(
        self, 
        inputs: Optional[torch.LongTensor] = None, # inference: with extended vocab ids (no unk id), 
        input_ids: Optional[torch.LongTensor] = None, # training: with extended vocab ids (no unk id)
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        
        inputs = inputs if inputs is not None else input_ids
        if self.training:
            self.eval()

        outputs = self.forward(input_ids=inputs)
        scores = outputs.logits
        
        # Greedy search
        output_ids = scores.argmax(dim=-1) # greedy search
        return output_ids
    
    def repetition_penalty(self, scores: torch.Tensor, inputs: torch.Tensor) -> torch.FloatTensor:
        penalty = 1
        score = torch.gather(scores, -1, inputs)
        score = score / penalty
        scores_processed = scores.scatter(-1, inputs, score)
        return scores_processed