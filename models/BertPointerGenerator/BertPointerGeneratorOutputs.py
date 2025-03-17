import torch
from transformers.modeling_outputs import ModelOutput, dataclass
from typing import Optional, Tuple

@dataclass
class BertPointerGeneratorOutputs(ModelOutput):
    """
    Base class for outputs of BertPointerGeneratorModel."
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            The Pointer Generator loss. Loss is required for Seq2SeqTrainer.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None