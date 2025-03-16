import torch
from transformers.modeling_outputs import Seq2SeqLMOutput, ModelOutput, dataclass
from typing import Optional, Tuple

@dataclass
class T5PointerGeneratorOutputs(ModelOutput):
    """
    Base class for outputs of T5PointerGeneratorModel."
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            The Pointer Generator loss. Loss is required for Seq2SeqTrainer.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_outputs: Optional[torch.FloatTensor] = None