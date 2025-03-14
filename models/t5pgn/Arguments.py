from utils import os
from transformers import Seq2SeqTrainingArguments, GenerationConfig
import datetime
from .T5PointerGeneratorTokenizer import T5PointerGeneratorTokenizer
from rouge import Rouge

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
        self.per_device_eval_batch_size : int = 32

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
    def __init__(self, tokenizer : T5PointerGeneratorTokenizer):
        self.rouge = Rouge()
        self.tokenizer = tokenizer

    def __call__(
        self, 
        eval_pred, 
        compute_result : bool = False, 
        full_result : bool = False
    ) -> dict:
        
        predictions, labels = eval_pred

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
        decoded_preds = self.tokenizer.batch_convert_extended_ids_to_tokens(predictions, local_vocabs=local_vocabs)
        decoded_labels = self.tokenizer.batch_convert_extended_ids_to_tokens(labels, local_vocabs=local_vocabs)

        # compute rouge
        hypothesis = [' '.join(pred) for pred in decoded_preds]
        reference = [' '.join(label) for label in decoded_labels]
        result = self.rouge.get_scores(hyps=hypothesis, refs=reference, avg=True)

        if compute_result:
            print('预测结果1：', ''.join(decoded_preds[0]))
            print('预测结果2：', ''.join(decoded_preds[1]))
            print('')

        if full_result:
            return result
        
        return {f'{key}.f': value['f'] for key, value in result.items()}