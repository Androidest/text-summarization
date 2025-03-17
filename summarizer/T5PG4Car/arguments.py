from utils import os
from transformers import Seq2SeqTrainingArguments, GenerationConfig
import datetime
from typing import Union, Dict

class T5PG4CarArguments(Seq2SeqTrainingArguments):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        curdir = os.path.dirname(__file__)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # extra
        self.random_seed = 1
        self.pretrained_path = 'pretrained/t5-small-chinese-cluecorpussmall'
        # self.pretrained_path = 'pretrained/Randeng-T5-784M-MultiTask-Chinese'
        self.data_use_B_dialogue=False
        self.data_use_B_question=False
        self.data_clean_workers=2
        self.data_preprocess_workers=3
        self.data_train_take = -1
        self.data_eval_take = 800

        # train
        data_size = 81718
        self.weight_decay = 0.01
        self.learning_rate = 1e-3
        self.per_device_train_batch_size : int = 10
        self.num_train_epochs : int = 5
        self.warmup_steps = data_size * self.num_train_epochs // self.per_device_train_batch_size // 10
        self.lr_scheduler_type="cosine"
        self.predict_with_generate=True
        self.generation_config = GenerationConfig()
        self.generation_config.max_length = 512
        self.generation_config.max_new_tokens = 128
        self.generation_config.repetition_penalty =  10.0

        # eval
        self.eval_strategy = "steps"
        self.eval_steps : int = 2000
        self.batch_eval_metrics = ["loss"] 
        self.per_device_eval_batch_size : int = 21

        # logging
        self.logging_dir = f"{curdir}/logs/{timestamp}"
        self.logging_steps = 50
        self.logging_strategy = "steps"

        # save
        self.output_dir = f"{curdir}/checkpoints/{timestamp}"
        self.save_strategy = "best"
        self.metric_for_best_model = "eval_loss"
        self.save_total_limit : int = 1
        self.load_best_model_at_end = True

