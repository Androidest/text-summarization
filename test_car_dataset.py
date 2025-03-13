from custom_datasets import *
from models.t5pgn import *
import os
import time
from models.t5pgn.T5PointerGeneratorTokenizer import T5PointerGeneratorTokenizer

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))

    pretrained_path = 'pretrained/t5-small-chinese-cluecorpussmall'
    tokenizer = T5PointerGeneratorTokenizer.from_pretrained()
    data_preprocessor = DataPreprocessorForT5PointerGenerator(tokenizer)

    dataset = CarSeq2SeqDataset(
        train=False, 
        clean_workers=5,
        use_B_dialogue=True, 
        use_B_question=False)

    print('Dataset size: ', len(dataset))

    time_start = time.time()
    dataset = dataset.map(data_preprocessor, workers=3)
    time_end = time.time()
    print(f'Dataset preprocessed in {time_end - time_start} seconds')

    for data in dataset:
        if data['extended_vocab_size'] > 0:
            print(data)
            break