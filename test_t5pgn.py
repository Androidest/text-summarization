from utils import *
from custom_datasets import *
import os
import pandas as pd
from models.t5pgn import *

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    data = pd.read_csv("custom_datasets/CarSeq2SeqDataset/data/dev.csv")
    summarizer = T5PointerGeneratorSummizer(
        "models/t5pgn/checkpoints/best2", 
        copy_weight=0.5, 
        device='cuda')

    # for i in range(50):
    #     print("-"*100)
    #     index = random.randint(0, len(data))
    #     text = data['Dialogue'][index]
    #     report = data['Report'][index]
    #     print('原文：', text, '\n')
    #     print('原摘要：', report, '\n')
    #     print('推理摘要：', summarizer.summarize(text), '\n')

    score = summarizer.evaluate(batch_size=8)

    print('Rouge Scores:')
    for key,val in score.items():
        print(key, val)