from utils import *
from custom_datasets import *
import os
import pandas as pd
from models.t5pgn import *

os.chdir(os.path.dirname(os.path.abspath(__file__)))

data = pd.read_csv("custom_datasets/CarSeq2SeqDataset/data/dev.csv")
summarizer = T5PointerGeneratorSummizer("models/t5pgn/checkpoints/best", device='cuda')

for i in range(10):
    print("-"*100)
    index = random.randint(0, len(data))
    text = data['Dialogue'][index]
    report = data['Report'][index]
    print('原文：', text, '\n')
    print('原摘要：', report, '\n')
    print('推理摘要：', summarizer.summarize(text), '\n')
