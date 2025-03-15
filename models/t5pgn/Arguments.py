from utils import os
from transformers import Seq2SeqTrainingArguments, GenerationConfig
import datetime
from .T5PointerGeneratorTokenizer import T5PointerGeneratorTokenizer
from rouge import Rouge
from typing import List, Tuple, Union, Dict
from torch import Tensor

class Arguments(Seq2SeqTrainingArguments):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        curdir = os.path.dirname(__file__)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # extra
        self.random_seed = 1
        self.pretrained_path = 'pretrained/t5-small-chinese-cluecorpussmall'
        # self.pretrained_path = 'pretrained/Randeng-T5-784M-MultiTask-Chinese'
        self.data_use_B_dialogue=False
        self.data_use_B_question=True
        self.data_clean_workers=5
        self.data_preprocess_workers=3

        # train
        data_size = 81718
        self.weight_decay = 0.01
        self.learning_rate = 8e-4
        self.per_device_train_batch_size : int = 8
        self.num_train_epochs : int = 4
        self.warmup_steps = 3000
        self.lr_scheduler_type="cosine"
        self.predict_with_generate=True
        self.generation_config = GenerationConfig()
        self.generation_config.max_length = 512
        self.generation_config.max_new_tokens = 128

        # eval
        self.eval_strategy = "steps"
        self.eval_steps : int = 2000
        self.batch_eval_metrics = ['loss'] 
        self.per_device_eval_batch_size : int = 6

        # logging
        self.logging_dir = f"{curdir}/logs/{timestamp}"
        self.logging_steps = 50
        self.logging_strategy = "steps"

        # save
        self.output_dir = f"{curdir}/checkpoints/{timestamp}"
        self.save_strategy = "best"
        self.metric_for_best_model = "eval_rouge-l.f" #"eval_loss"
        self.save_total_limit : int = 1
        self.load_best_model_at_end = True

class RougeMetric:
    def __init__(
        self, 
        tokenizer : T5PointerGeneratorTokenizer, 
        full_result : bool = False
    ):
        self.rouge = Rouge()
        self.tokenizer = tokenizer
        self.full_result = full_result

    def __call__(
        self, 
        eval_pred : Tuple[Tensor, Tensor],
        compute_result : bool = False, # used by Seq2SeqTrainer, true on last batch of eval
        local_vocabs : List[dict] = None
    ) -> dict:
        
        predictions, labels = eval_pred

        if local_vocabs is None:
            local_vocabs = []
            for (pred, label) in zip(predictions, labels):
                local_vocab = {}
                for token_id in pred + label:
                    token_id = token_id.item()
                    if token_id >= self.tokenizer.vocab_size:
                        num = token_id - self.tokenizer.vocab_size
                        local_vocab[f'[UNK{num}]'] = token_id
                local_vocabs.append(local_vocab)

        # ids to tokens
        decoded_preds = self.tokenizer.batch_decode_extended_ids(predictions, local_vocabs=local_vocabs, join_space=" ")
        decoded_labels = self.tokenizer.batch_decode_extended_ids(labels, local_vocabs=local_vocabs, join_space=" ")

        # compute rouge
        avg = not self.full_result
        result = self.rouge.get_scores(hyps=decoded_preds, refs=decoded_labels, avg=avg)

        if compute_result:
            print('预测结果1：', decoded_preds[0].replace(' ', ''))
            print('预测结果2：', decoded_preds[1].replace(' ', ''))
            print('')

        if self.full_result:
            return result
        
        return {f'{key}.f': value['f'] for key, value in result.items()}