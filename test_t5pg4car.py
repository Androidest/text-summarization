from utils import *
from custom_datasets.CarSeq2SeqDataset import *
from summarizer.T5PG4Car import *
import pandas as pd
import os
import time
from rouge import Rouge

os.chdir(os.path.dirname(__file__))

def test_rouge():
    reference_tokens = ["今", "日", "天", "气", "很", "好"]
    hypothesis_tokens = ["今", "天", "的", "天", "气", "不", "错"]

    reference_str = ' '.join(reference_tokens)
    hypothesis_str = ' '.join(hypothesis_tokens)

    rouge = Rouge()
    scores = rouge.get_scores(hypothesis_str, reference_str, avg=True, ignore_empty=True)
    print(scores)

def test_dataset():
    args = T5PG4CarArguments()
    tokenizer = T5PointerGeneratorTokenizer.from_pretrained(args.pretrained_path)

    print('unk_token_id', tokenizer.unk_token_id)
    print('bos_token_id', tokenizer.bos_token_id)
    print('eos_token_id', tokenizer.eos_token_id)
    print('pad_token_id', tokenizer.pad_token_id)

    dataset = CarSeq2SeqDataset(
        train=False, 
        clean_workers=5,
        use_B_dialogue=False, 
        use_B_question=True,
        take=-1
    )
    dataset.map(T5PG4CarDataPreprocessor(tokenizer, args.generation_config))
    
    count = 0
    for data in dataset:
        if data['extended_vocab_size'] > 5:
            print(data)
            count += 1
            if count > 5:
                break

def test_summarizer(path):
    data = pd.read_csv("custom_datasets/CarSeq2SeqDataset/data/dev.csv")
    summarizer = T5PG4CarSummizer.from_pretrained(path)
    summarizer = summarizer.to("cuda")

    use_rand = True
    selected_data = [10134, 2076, 3966, 9098, 9107, 3952, 10842, 4422]
    if use_rand:
        selected_data = [random.randint(0, len(data)) for _ in range(30)]

    for i in selected_data:
        print("-"*100)
        text = data['Dialogue'][i]
        report = data['Report'][i]
        print(f'[{i}]原文：', text, '\n')
        print(f'[{i}]原摘要：', report, '\n')
        print('推理摘要：', summarizer.summarize(text), '\n')

def test_summizer_evaluate(path):
    summarizer = T5PG4CarSummizer.from_pretrained(path)
    summarizer = summarizer.to("cuda")
    score = summarizer.evaluate(batch_size=20, take=5000)

    print('Rouge Scores:')
    for key,val in score.items():
        print(key, val)

if __name__ == '__main__':
    # test_rouge()
    # test_dataset()
    
    model_path = "summarizer/T5PG4Car/checkpoints/best1"
    test_summarizer(model_path)
    test_summizer_evaluate(model_path)