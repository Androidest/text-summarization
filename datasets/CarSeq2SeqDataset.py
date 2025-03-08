from torch.utils.data import Dataset
import pandas as pd
import os
import re
from multiprocessing import Pool
import numpy as np
import time
import math

SEP = '<SEP>'

class CarDataCleaner:

    @ staticmethod
    def clean_dataset(
        src : str, 
        dist : str, 
        core : int = 1, 
        use_B_dialogue : bool = True, 
        use_B_question : bool = False
    ):
        """
        多进程清洗数据集
        清洗后，对话中的角色：
            A: 技师
            B: 车主

        Args:
            src (str): 原始数据集文件路径
            dist (str): 清洗后的数据集文件保存路径
            core (int, optional): 多进程的核心数. 默认 1.
            use_B_dialogue (bool, optional): 是否使用 车主 的对话. 默认 True.
            use_B_question (bool, optional): 是否在对话前添加 车主 的问题. 默认 False.
        """

        # 读取数据, 并按照core的数量进行分割
        dataframe = pd.read_csv(src, encoding='utf8')
        cleaned_data = []

        # 多进程清洗数据
        with Pool(core) as pool:
            partition_size = math.ceil(len(dataframe) / core)
            data_partition = []
            for i in range(core):
                params = (
                    dataframe[i * partition_size : (i + 1) * partition_size],
                    use_B_dialogue,
                    use_B_question
                )
                data_partition.append(params)
            data_partition = pool.starmap(CarDataCleaner.clean_func, data_partition)
            cleaned_data = sum(data_partition, [])

        # 合进程程数据
        # 保存数据
        with open(dist, 'w', encoding='utf8') as f:
            f.writelines(cleaned_data)

    @ staticmethod
    def clean_func(df : pd.DataFrame, use_B_dialogue : bool, use_B_question : bool):
        data = []
        for index, row in df.iterrows():
            question, dialog, report = row.Question, row.Dialogue, row.Report

            if not isinstance(dialog, str):
                continue

            # dialog
            dialog = CarDataCleaner.dialog_cleaning(dialog)
            if not use_B_dialogue:
                dialog = CarDataCleaner.remove_B_dialogue(dialog)
            if use_B_question:
                dialog = f'B:{question}{dialog}'
            dialog = CarDataCleaner.punctuation_cleaning(dialog)

            if len(dialog) == 0:  # 若清洗后对话为空，则跳过（说明技师的发的全是语音和图片，并且清除了车主的对话）
                continue

            # report
            if not isinstance(report, str):
                report = '随时联系。'
            else:
                report = CarDataCleaner.punctuation_cleaning(report)
            
            line = f'{dialog}{SEP}{report}\n'
            data.append(line)

        return data
    
    @ staticmethod
    def remove_B_dialogue(text : str):
        text = re.sub(r'B：.*?[\|]', '', text) # 删除所有B的对话（车主）
        return text

    @ staticmethod
    def dialog_cleaning(text : str):
        text = re.sub(r'\[语音\]', '', text)
        text = re.sub(r'\[图片\]', '', text)
        text = re.sub(r'技师说', 'A', text)
        text = re.sub(r'车主说', 'B', text)
        text = re.sub(r'A：\|', '', text) # 删除语音和图片后，把留下空的对话信息删除，如 |A：|
        text = re.sub(r'B：\|', '', text) # 删除语音和图片后，把留下空的对话信息删除，如 |B：|
        text = re.sub(r'A：$', '', text) 
        text = re.sub(r'B：$', '', text) 
        return text

    @ classmethod
    def punctuation_cleaning(cls, text : str):
        text = text.strip()
        text = re.sub(r'(\s)?[\。](\s)?', '。', text) 
        text = re.sub(r'(\s)?[\,\，](\s)?', '，', text)
        text = re.sub(r'(\s)?[\?\？](\s)?', '？', text) 
        text = re.sub(r'(\s)?[\!\！](\s)?', '！', text) 
        text = re.sub(r'(\s)?[\:](\s)?', ':', text) 
        text = re.sub(r'(\s)+', '，', text) 

        text = re.sub(r'[\。\，\？\！]?\|[\。\，\？\！]?', '。', text) 
        text = re.sub(r'[\。\，\？\！]?\。[\。\，\？\！]?', '。', text) 
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\)\）\：]$', '。', text) # \u4e00-\u9fa5a 代表中文
        text = text + '。' if re.search(r'[\u4e00-\u9fa5a-zA-Z0-9]$', text) else text # 如果最后一个字符是文本字符，则添加句号
        return text

class CarSeq2SeqDataset(Dataset):
    def __init__(self, 
            path: str, 
            config: object = None, 
            force_clean: bool = False,
            clean_workers: int = 1,
            use_B_dialogue : bool = True, 
            use_B_question : bool = False
        ):
        self.data = []
        # self.tokenizer = config.model_tokenizer
        filename, ext = os.path.splitext(path)
        temp_filename = f'{filename}.temp.text'

        if force_clean or not os.path.exists(temp_filename):
            CarDataCleaner.clean_dataset(path, temp_filename, clean_workers, use_B_dialogue, use_B_question)
        
        with open(temp_filename, 'r', encoding='utf8') as f:
            for line in f:
                x, y = line.strip().split(SEP)
                # x = self.tokenizer(x)
                self.data.append((x, y))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i) -> iter:
        return self.data[i]

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    time_start = time.time()
    dataset = CarSeq2SeqDataset(
        '../data/test.csv', 
        force_clean=True, 
        clean_workers=5,
        use_B_dialogue=True, 
        use_B_question=False)
    
    time_end = time.time()
    x, y = dataset[0]
    print('time cost', time_end - time_start, 's')
    print(x, '\n')
    print(y)