from models.T5PointerGenerator import *
from rouge import Rouge
from torch import Tensor
from typing import List, Tuple

class T5PG4CarRougeMetric:
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
        decoded_preds = self.tokenizer.batch_decode_extended_ids(
            predictions, 
            local_vocabs=local_vocabs, 
            join_space=" ", 
            remove_special_tokens=False)
        
        decoded_labels = self.tokenizer.batch_decode_extended_ids(
            labels, 
            local_vocabs=local_vocabs, 
            join_space=" ",
            remove_special_tokens=False)
        
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