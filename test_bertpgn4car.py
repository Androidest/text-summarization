from utils import *
from custom_datasets.CarSeq2SeqDataset import *
from summarizer.BertPGN4Car import *
import pandas as pd
import os
import time
from rouge import Rouge

os.chdir(os.path.dirname(__file__))

def test_dataset():
    args = BertPGN4CarArguments()
    tokenizer = BertPointerGeneratorTokenizer.from_pretrained(args.pretrained_path)

    dataset = CarSeq2SeqDataset(
        train=False, 
        clean_workers=5,
        use_B_dialogue=False, 
        use_B_question=True,
        take=-1
    )
    dataset.map(BertPGN4CarDataPreprocessor(tokenizer, args.generation_config))
    
    count = 0
    for data in dataset:
        if data['extended_vocab_size'] > 5:
            print(data)
            count += 1
            if count > 5:
                break

def test_summarizer(path):
    data = pd.read_csv("custom_datasets/CarSeq2SeqDataset/data/dev.csv")
    summarizer = BertPGN4CarSummizer.from_pretrained(path)
    summarizer = summarizer.to("cuda")

    use_rand = True
    selected_data = [10134, 2076, 3966, 9098, 9107, 3952, 10842, 4422]
    if use_rand:
        selected_data = [random.randint(0, len(data)) for _ in range(30)]

    for i in selected_data:
        print("-"*100)
        text = data['Dialogue'][i]
        report = data['Report'][i]
        # print(f'[{i}]原文：', text, '\n')
        print(f'[{i}]原摘要：', report, '\n')
        print('推理摘要：', summarizer.summarize(text), '\n')

def test_summizer_evaluate(path):
    summarizer = BertPGN4CarSummizer.from_pretrained(path)
    summarizer = summarizer.to("cuda")
    score = summarizer.evaluate(batch_size=20, take=-1)

    print('Rouge Scores:')
    for key,val in score.items():
        print(key, val)

if __name__ == '__main__':
    # test_dataset()
    
    model_path = "summarizer/T5PG4Car/checkpoints/best1"
    test_summarizer(model_path)
    test_summizer_evaluate(model_path)