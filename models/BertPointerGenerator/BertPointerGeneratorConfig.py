from transformers import BertConfig

class BertPointerGeneratorConfig(BertConfig):
    model_type = "bert"
    def __init__(
            self, 
            unk_token_id : int = None,
            cls_token_id : int = None,
            eos_token_id : int = None,
            max_len : int = 512,
            **kwargs):
        super().__init__(**kwargs)
        self.unk_token_id = unk_token_id
        self.cls_token_id = cls_token_id
        self.eos_token_id = eos_token_id
        self.max_len = max_len