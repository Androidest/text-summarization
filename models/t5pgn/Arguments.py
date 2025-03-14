from utils import os
from transformers import Seq2SeqTrainingArguments, GenerationConfig
import datetime

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
        self.eval_steps : int = 10
        self.batch_eval_metrics = ['loss'] #['loss', 'rouge1', 'rouge2', 'rougeL']
        self.per_device_eval_batch_size : int = 32

        # logging
        self.logging_dir = f"{curdir}/logs/{timestamp}"
        self.logging_steps = 50
        self.logging_strategy = "steps"

        # save
        self.output_dir = f"{curdir}/checkpoints/{timestamp}"
        # self.save_strategy = "best"
        # self.metric_for_best_model = "eval_loss"
        self.save_total_limit : int = 1
        # self.load_best_model_at_end = True

    @staticmethod
    def compute_metrics(eval_pred, compute_result : bool):
        if compute_result:
            print(eval_pred)
        return {'loss': 10.0 }