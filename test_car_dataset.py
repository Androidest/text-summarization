from custom_datasets import *
from models.t5pgn import *
import os
import time

from rouge import Rouge

reference_tokens = ["今", "日", "天", "气", "很", "好"]
hypothesis_tokens = ["今", "天", "的", "天", "气", "不", "错"]

# 将token列表转换为字符串
reference_str = ' '.join(reference_tokens)
hypothesis_str = ' '.join(hypothesis_tokens)

# 初始化Rouge
rouge = Rouge()

# 计算ROUGE分数
scores = rouge.get_scores([hypothesis_str, hypothesis_str], [reference_str, reference_str], avg=False)
print(scores)

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))

    args = Arguments()
    config = T5Config.from_pretrained(args.pretrained_path)
    model = T5PointerGeneratorModel.from_pretrained(args.pretrained_path)
    tokenizer = T5PointerGeneratorTokenizer.from_pretrained(args.pretrained_path)

    print('unk_token_id', tokenizer.unk_token_id)
    print('bos_token_id', tokenizer.bos_token_id)
    print('eos_token_id', tokenizer.eos_token_id)
    print('pad_token_id', tokenizer.pad_token_id)

    dataset = CarSeq2SeqDataset(
        train=False, 
        clean_workers=5,
        use_B_dialogue=False, 
        use_B_question=True)
    
    print('Dataset size: ', len(dataset))
    data_preprocessor = DataPreprocessorForT5PointerGenerator(tokenizer, args.generation_config)
    dataset = dataset.take(10)
    dataset = dataset.map(data_preprocessor, workers=3)
    print('Dataset size: ', len(dataset))

    for data in dataset:
        if data['extended_vocab_size'] > 0:
            print(data)
            break

    
