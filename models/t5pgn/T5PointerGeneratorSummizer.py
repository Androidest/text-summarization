from .T5PointerGeneratorModel import *
from .T5PointerGeneratorTokenizer import *
from .Arguments import RougeMetric, Arguments
from custom_datasets import *
from transformers import GenerationConfig
from typing import Optional, Tuple, Union
import torch
from tqdm import tqdm

class T5PointerGeneratorSummizer:
    def __init__(
            self, path : str, 
            copy_weight : float = 1,  # weight of copy: final_p_gen = p_gen * copy_weight
            device : str = "cpu"
        ):
        self.copy_weight = copy_weight 
        self.model = T5PointerGeneratorModel.from_pretrained(path)
        self.tokenizer = T5PointerGeneratorTokenizer.from_pretrained(path)
        self.generation_config = GenerationConfig.from_pretrained(path)
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.rouge_metrics = RougeMetric(self.tokenizer, full_result=True)

        args = Arguments()
        self.data_use_B_dialogue = args.data_use_B_dialogue
        self.data_use_B_question = args.data_use_B_question
        self.data_clean_workers = args.data_clean_workers
        self.data_preprocess_workers = args.data_preprocess_workers
        self.eval_dataset = None

    def summarize(self, input_text : str) -> str:
        # clean text
        CarDataCleaner.clean_dialogue(input_text, self.data_use_B_dialogue)

        # encode text
        input_ids, local_vocab = self.tokenizer.encode_extended_ids(
            input_text,
            max_length=self.generation_config.max_length,
            return_tensors=True,
            add_special_tokens=False
        )

        # generate summary
        input_ids = input_ids.unsqueeze(0).to(self.device) # to batch (1, seq_len) 
        output_ids = self.model.generate(
            input_ids, 
            extended_vocab_size=len(local_vocab), 
            copy_weight = self.copy_weight # weight of copy, for inference only: final_p_gen = p_gen * copy_weight
        )
        
        # decode summary
        summirized_text = self.tokenizer.decode_extended_ids(output_ids[0], local_vocab)
        return summirized_text
    
    def evaluate(self, batch_size : int = 8, dataset : CarSeq2SeqDataset = None) -> dict:
        # prepare eval dataset
        if dataset is None:
            if self.eval_dataset is None:
                self.eval_dataset = CarSeq2SeqDataset(
                    train=False,
                    clean_workers=self.data_clean_workers,
                    use_B_dialogue=self.data_use_B_dialogue, 
                    use_B_question=self.data_use_B_question,
                )
                data_preprocessor = DataPreprocessorForT5PointerGenerator(self.tokenizer, self.generation_config)
                self.eval_dataset.map(data_preprocessor, workers=self.data_preprocess_workers)
            dataset = self.eval_dataset

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DataCollatorForT5PointerGenerator(self.tokenizer)
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
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    extended_vocab_size=extended_vocab_size)
                
                scores = self.rouge_metrics((predictions, labels), local_vocabs=local_vocabs)
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
        
