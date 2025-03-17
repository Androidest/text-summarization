from utils import *
from custom_datasets.CarSeq2SeqDataset import *
from models.BertPointerGenerator import *
from transformers import Seq2SeqTrainer
from .metrics import *
from .data_preprocess import *
from .arguments import *
from typing import Optional, Tuple, Union
import torch
from tqdm import tqdm

class BertPGN4CarSummizer:
    def __init__(self, 
        model : BertPointerGeneratorModel = None, 
        tokenizer : BertPointerGeneratorTokenizer = None, 
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.args = BertPGN4CarArguments()

        if self.model:
            self.model.eval()
            self.model.to("cpu")

    @classmethod
    def from_pretrained(cls, path : str):
        config = BertPointerGeneratorConfig.from_pretrained(path)
        return cls(
            model=BertPointerGeneratorModel.from_pretrained(path, config),
            tokenizer=BertPointerGeneratorTokenizer.from_pretrained(path),
        )

    def to(self, device : str):
        self.device = device
        self.model.to(device)
        return self

    def train(self, args : Optional[BertPGN4CarArguments] = None):
        if args:
            self.args = args

        set_seed(self.args.random_seed)

        print("======== Loading Pretrained Model =======")
        print(f'Loading pretrained model from {self.args.pretrained_path}')
        # Load pretrained model and tokenizer
        config = BertPointerGeneratorConfig.from_pretrained(self.args.pretrained_path)
        model = BertPointerGeneratorModel.from_pretrained(self.args.pretrained_path, config=config)
        tokenizer = BertPointerGeneratorTokenizer.from_pretrained(self.args.pretrained_path)
        metrics = BertPGN4CarRougeMetric(tokenizer)

        # sync generation_config and tokenizer
        config.cls_token_id = tokenizer.cls_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.unk_token_id = tokenizer.unk_token_id

        # DataPreprocessor: Process tokenization, generate input_ids, etc
        data_preprocessor = BertPGN4CarDataPreprocessor(tokenizer, config) 
        # DataCollator: Handle dynamic padding, generate attention masks, etc
        data_collator = BertPGN4CarDataCollator(config) 

        print("======== Preparing datasets =======")
        # Load datasets and preprocess them
        train_dataset = CarSeq2SeqDataset(
            train=True,
            clean_workers=self.args.data_clean_workers,
            use_B_dialogue=self.args.data_use_B_dialogue, 
            use_B_question=self.args.data_use_B_question,
            take=self.args.data_train_take
        )
        train_dataset.map(data_preprocessor, workers=self.args.data_preprocess_workers)

        eval_dataset = CarSeq2SeqDataset(
            train=False,
            clean_workers=self.args.data_clean_workers,
            use_B_dialogue=self.args.data_use_B_dialogue, 
            use_B_question=self.args.data_use_B_question,
            take=self.args.data_eval_take
        )
        eval_dataset.map(data_preprocessor, workers=self.args.data_preprocess_workers)

        print("======== Start training =======")
        # Start training
        trainer = Seq2SeqTrainer(
            model=model,
            args=self.args,
            processing_class=tokenizer.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=metrics,
        )
        trainer.train()
        print("======== Finish training =======")

        # Save best model
        trainer.evaluate()
        path = f'{self.args.output_dir}/best'
        trainer.save_model(path)
        print(f'Best model saved to {path}')

    def evaluate(
            self, 
            batch_size : int = 8, 
            take : int = -1 # -1 complete dataset, 100 mean take first 100 samples
        ) -> dict:

        rouge_metrics = BertPGN4CarRougeMetric(self.tokenizer, full_result=True)
        data_preprocessor = BertPGN4CarDataPreprocessor(self.tokenizer, self.model.config)
        data_collator = BertPGN4CarDataCollator(self.tokenizer, return_local_vocab=True)
        
        # prepare eval dataset
        eval_dataset = CarSeq2SeqDataset(
            train=False,
            clean_workers=self.args.data_clean_workers,
            use_B_dialogue=self.args.data_use_B_dialogue, 
            use_B_question=self.args.data_use_B_question,
            take=take
        )
        eval_dataset.map(data_preprocessor, workers=self.args.data_preprocess_workers)

        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )

        # evaluate
        all_scores = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating Summarizer"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                extended_vocab_size = batch['extended_vocab_size']
                local_vocabs = batch['local_vocabs']
                labels = batch['labels'].to(self.device)

                predictions = self.model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    extended_vocab_size=extended_vocab_size,
                )
                
                scores = rouge_metrics((predictions, labels), local_vocabs=local_vocabs)
                all_scores.extend(scores)

        score = all_scores[0]
        for i in range(1, len(all_scores)):
            for rouge_type, rouge_value in all_scores[i].items():
                for key, val in rouge_value.items():
                    score[rouge_type][key] += val
        
        for rouge_type, rouge_value in score.items():
            for key, val in rouge_value.items():
                score[rouge_type][key] /= len(all_scores)

        return score
        
    def summarize(self, input_text : str) -> str:
        # clean text
        input_text = CarDataCleaner.clean_dialogue(input_text, self.args.data_use_B_dialogue)

        # encode text
        input_ids, local_vocab = self.tokenizer.encode_extended_ids(
            input_text,
            max_length=self.model.config.max_length,
            return_tensors=True,
            add_special_tokens=True,
        )

        # generate summary
        input_ids = input_ids.unsqueeze(0).to(self.device) # to batch (1, seq_len) 
        output_ids = self.model.generate(
            inputs = input_ids, 
            extended_vocab_size=len(local_vocab), 
        )
        
        # decode summary
        summirized_text = self.tokenizer.decode_extended_ids(output_ids[0], local_vocab)
        return summirized_text
    
