from .Arguments import Arguments, RougeMetric
from .T5PointerGeneratorTokenizer import *
from .T5PointerGeneratorTokenizer import *
from .T5PointerGeneratorModel import *
from .T5PointerGeneratorSummizer import *
from custom_datasets import CarSeq2SeqDataset
from utils import *
from transformers import Seq2SeqTrainer, T5Config

def create_trainer():
    args = Arguments()
    set_seed(args.random_seed)

    # Load pretrained model and tokenizer
    config = T5Config.from_pretrained(args.pretrained_path)
    model = T5PointerGeneratorModel.from_pretrained(args.pretrained_path)
    tokenizer = T5PointerGeneratorTokenizer.from_pretrained(args.pretrained_path)
    metrics = RougeMetric(tokenizer)

    # DataPreprocessor: Process tokenization, generate input_ids, etc
    data_preprocessor = DataPreprocessorForT5PointerGenerator(tokenizer, args.generation_config) 
    # DataCollator: Handle dynamic padding, generate attention masks, etc
    data_collator = DataCollatorForT5PointerGenerator(config) 

    # Load datasets and preprocess them
    train_dataset = CarSeq2SeqDataset(
        train=True,
        clean_workers=args.data_clean_workers,
        use_B_dialogue=args.data_use_B_dialogue, 
        use_B_question=args.data_use_B_question)
    train_dataset.take(100)
    train_dataset.map(data_preprocessor, workers=args.data_preprocess_workers)

    eval_dataset = CarSeq2SeqDataset(
        train=False,
        clean_workers=args.data_clean_workers,
        use_B_dialogue=args.data_use_B_dialogue, 
        use_B_question=args.data_use_B_question)
    eval_dataset.take(10)
    eval_dataset.map(data_preprocessor, workers=args.data_preprocess_workers)

    # Create Trainer instance
    return Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=metrics,
    )