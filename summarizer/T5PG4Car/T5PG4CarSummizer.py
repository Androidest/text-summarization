from utils import *
from custom_datasets.CarSeq2SeqDataset import *
from models.T5PointerGenerator import *
from transformers import GenerationConfig, Seq2SeqTrainer, T5Config
from .metrics import *
from .data_preprocess import *
from .arguments import *
from typing import Optional, Tuple, Union
import torch
from tqdm import tqdm

class T5PG4CarSummizer:
    def __init__(self, 
        model : T5PointerGeneratorModel = None, 
        tokenizer : T5PointerGeneratorTokenizer = None, 
        generation_config : GenerationConfig = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.args = T5PG4CarArguments()

        if self.model:
            self.model.eval()
            self.model.to("cpu")

    @classmethod
    def from_pretrained(cls, path : str):
        return cls(
            model=T5PointerGeneratorModel.from_pretrained(path),
            tokenizer=T5PointerGeneratorTokenizer.from_pretrained(path),
            generation_config=GenerationConfig.from_pretrained(path),
        )

    def to(self, device : str):
        self.device = device
        self.model.to(device)
        return self

    def train(self, args : Optional[T5PG4CarArguments] = None):
        if args:
            self.args = args

        set_seed(self.args.random_seed)

        print("======== Loading Pretrained Model =======")
        print(f'Loading pretrained model from {self.args.pretrained_path}')
        # Load pretrained model and tokenizer
        config = T5Config.from_pretrained(self.args.pretrained_path)
        model = T5PointerGeneratorModel.from_pretrained(self.args.pretrained_path)
        tokenizer = T5PointerGeneratorTokenizer.from_pretrained(self.args.pretrained_path)
        metrics = T5PG4CarRougeMetric(tokenizer)

        # sync generation_config and tokenizer
        self.args.generation_config.bos_token_id = tokenizer.bos_token_id
        self.args.generation_config.eos_token_id = tokenizer.eos_token_id
        self.args.generation_config.pad_token_id = tokenizer.pad_token_id

        # DataPreprocessor: Process tokenization, generate input_ids, etc
        data_preprocessor = T5PG4CarDataPreprocessor(tokenizer, self.args.generation_config) 
        # DataCollator: Handle dynamic padding, generate attention masks, etc
        data_collator = T5PG4CarDataCollator(config) 

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

        rouge_metrics = T5PG4CarRougeMetric(self.tokenizer, full_result=True)
        data_preprocessor = T5PG4CarDataPreprocessor(self.tokenizer, self.generation_config)
        data_collator = T5PG4CarDataCollator(self.tokenizer, return_local_vocab=True)
        
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
                    compute_last_token=True,
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
        
    def summarize(self, input_text : str, copy_weight : int = 1) -> str:
        # clean text
        input_text = CarDataCleaner.clean_dialogue(input_text, self.args.data_use_B_dialogue)

        # encode text
        input_ids, local_vocab = self.tokenizer.encode_extended_ids(
            input_text,
            max_length=self.generation_config.max_length,
            return_tensors=True,
            add_special_tokens=False,
        )

        # generate summary
        input_ids = input_ids.unsqueeze(0).to(self.device) # to batch (1, seq_len) 
        output_ids = self.model.generate(
            inputs = input_ids, 
            extended_vocab_size=len(local_vocab), 
            copy_weight = copy_weight, # weight of copy, for inference only: final_p_gen = p_gen * copy_weight
            compute_last_token=True,
        )
        
        # decode summary
        summirized_text = self.tokenizer.decode_extended_ids(output_ids[0], local_vocab)
        return summirized_text
    
