from custom_datasets import *
from models.t5pgn import *
import os
import time

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))

    pretrained_path = 'pretrained/t5-small-chinese-cluecorpussmall'
    config = T5Config.from_pretrained(pretrained_path)
    model = T5PointerGeneratorModel.from_pretrained(pretrained_path)
    tokenizer = T5PointerGeneratorTokenizer.from_pretrained(pretrained_path)

    dataset = CarSeq2SeqDataset(
        train=False, 
        clean_workers=5,
        use_B_dialogue=False, 
        use_B_question=True)
    
    print('Dataset size: ', len(dataset))
    data_preprocessor = DataPreprocessorForT5PointerGenerator(tokenizer)
    dataset = dataset.take(10)
    dataset = dataset.map(data_preprocessor, workers=3)
    print('Dataset size: ', len(dataset))

    for data in dataset:
        if data['extended_vocab_size'] > 0:
            print(data)
            break

    
