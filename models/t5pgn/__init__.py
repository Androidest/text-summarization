from .T5PointerGeneratorTokenizer import *
from .T5PointerGeneratorTokenizer import *
from .T5PointerGeneratorModel import *
from custom_datasets import CarSeq2SeqDataset
from utils import *
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Config

class T5PointerGeneratorTrainingArgs(Seq2SeqTrainingArguments):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        curdir = os.path.dirname(__file__)

        # train
        data_size = 81718
        self.weight_decay = 0.01
        self.learning_rate = 8e-5
        self.per_device_train_batch_size : int = 8
        self.num_train_epochs : int = 5
        epoch_steps = data_size // self.per_device_train_batch_size
        self.warmup_steps = 1000
        self.lr_scheduler_type="cosine"
        # extra
        self.random_seed = 1
        self.pretrained_path = 'pretrained/t5-small-chinese-cluecorpussmall'
        # self.pretrained_path = 'pretrained/Randeng-T5-784M-MultiTask-Chinese'
        self.data_use_B_dialogue=True
        self.data_use_B_question=False
        self.data_clean_workers=5
        self.data_preprocess_workers=3
        # logging
        self.logging_dir = f"{curdir}/logs"
        self.logging_steps = 50
        self.logging_strategy = "steps"
        # save
        self.output_dir = f"{curdir}/checkpoints"
        self.save_strategy = "best"
        self.metric_for_best_model = "eval_loss"
        self.save_total_limit : int = 1
        self.load_best_model_at_end = True
        # eval
        self.eval_strategy = "steps"
        self.eval_steps : int = 1000
        self.batch_eval_metrics = ['loss'] #['loss', 'rouge1', 'rouge2', 'rougeL']
        self.per_device_eval_batch_size : int = 45
        
def create_trainer():
    args = T5PointerGeneratorTrainingArgs()
    set_seed(args.random_seed)

    # Load pretrained model and tokenizer
    config = T5Config.from_pretrained(args.pretrained_path)
    model = T5PointerGeneratorModel.from_pretrained(args.pretrained_path)
    tokenizer = T5PointerGeneratorTokenizer.from_pretrained(args.pretrained_path, legacy=True)

    # DataPreprocessor: Process tokenization, generate input_ids, etc
    data_preprocessor = DataPreprocessorForT5PointerGenerator(tokenizer) 
    # DataCollator: Handle dynamic padding, generate attention masks, etc
    data_collator = DataCollatorForT5PointerGenerator(config) 

    # Load datasets and preprocess them
    train_dataset = CarSeq2SeqDataset(
            train=True,
            clean_workers=args.data_clean_workers,
            use_B_dialogue=args.data_use_B_dialogue, 
            use_B_question=args.data_use_B_question
        ).map(data_preprocessor, workers=args.data_preprocess_workers)

    eval_dataset = CarSeq2SeqDataset(
            train=False,
            clean_workers=args.data_clean_workers,
            use_B_dialogue=args.data_use_B_dialogue, 
            use_B_question=args.data_use_B_question
        ).map(data_preprocessor, workers=args.data_preprocess_workers)

    # Create Trainer instance
    return Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )