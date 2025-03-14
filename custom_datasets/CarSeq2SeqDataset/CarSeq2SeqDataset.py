import pandas as pd
import os
import re
from multiprocessing import Pool
import math
from torch.utils.data import Dataset

SEP = '<SEP>'
A = '师'
B = '主'

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
        if core == 1:
            cleaned_data = CarDataCleaner.clean_func(dataframe, use_B_dialogue, use_B_question)
        else:
            print(f'Using {core} cores to clean data {src}')
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
            print(f'Cleaned data {src}')

        # 合进程程数据
        # 保存数据
        with open(dist, 'w', encoding='utf8') as f:
            f.writelines(cleaned_data)

    @staticmethod
    def clean_dialogue(dialog : str, use_B_dialogue : bool) -> str:
        dialog = CarDataCleaner.dialog_basic_cleaning(dialog)
        if not use_B_dialogue:
            dialog = CarDataCleaner.remove_B_dialogue(dialog)
        dialog = CarDataCleaner.punctuation_cleaning(dialog)
        dialog = CarDataCleaner.replaceAB(dialog)
        return dialog
    
    @ staticmethod
    def clean_func(df : pd.DataFrame, use_B_dialogue : bool, use_B_question : bool):
        data = []
        for index, row in df.iterrows():
            question, dialog, report = row.Question, row.Dialogue, row.Report

            if not isinstance(dialog, str):
                continue

            # dialog
            dialog = CarDataCleaner.dialog_basic_cleaning(dialog)
            if not use_B_dialogue:
                dialog = CarDataCleaner.remove_B_dialogue(dialog)
            if use_B_question:
                dialog = f'B:{question}{dialog}'
            dialog = CarDataCleaner.punctuation_cleaning(dialog)
            dialog = CarDataCleaner.replaceAB(dialog)

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
    def replaceAB(text : str):
        text = re.sub(r'A：', f'{A}：', text) 
        text = re.sub(r'B：', f'{B}：', text)
        return text
    
    @ staticmethod
    def remove_B_dialogue(text : str):
        text = re.sub(r'B：.*?[\|]', '', text) # 删除所有B的对话（车主）
        return text

    @ staticmethod
    def dialog_basic_cleaning(text : str):
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
            train: bool = True,
            force_clean: bool = False,
            clean_workers: int = 1,
            use_B_dialogue : bool = True, 
            use_B_question : bool = False,
            data_path : str = None,
            take : int = -1
        ):
        curdir = os.path.dirname(__file__)
        path = f'{curdir}/data/train.csv' if train else f'{curdir}/data/dev.csv'
        if data_path:
            path = data_path

        filename, ext = os.path.splitext(path)
        temp_filename = f'{filename}.temp.text'

        if force_clean or not os.path.exists(temp_filename):
            CarDataCleaner.clean_dataset(path, temp_filename, clean_workers, use_B_dialogue, use_B_question)
        
        self._data = []
        with open(temp_filename, 'r', encoding='utf8') as f:
            for i, line in enumerate(f):
                if take > 0 and i >= take:
                    break
                x, y = line.strip().split(SEP)
                self._data.append({ 'x': x, 'y': y })

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i) -> iter:
        return self._data[i]

    def map(self, callback, workers : int = 1):
        print(f'Pre-processing data with {workers} cores')
        
        if workers == 1:
            self._data = list(map(callback, self._data))
        else:
            with Pool(workers) as pool:
                partition_size = math.ceil(len(self._data) / workers)
                
                data_partition = []
                for i in range(workers):
                    data = self._data[i * partition_size : (i + 1) * partition_size]
                    data_partition.append((callback, data))

                data_partition = pool.starmap(CarSeq2SeqDataset.filter_func, data_partition)
                self._data = sum(data_partition, [])

        print('Mapping finished')
        return self

    @ staticmethod
    def filter_func(callback, data : list):
        return list(map(callback, data))
    
    def take(self, n : int):
        self._data = self._data[:n]
        return self
    
if __name__ == "__main__":
    import time
    from torch.utils.data import DataLoader

    time_start = time.time()
    dataset = CarSeq2SeqDataset(
        train=False, 
        force_clean=True, 
        clean_workers=5,
        use_B_dialogue=True, 
        use_B_question=False)
    
    dataset.map(lambda x: {
        'input_ids': x['x'],
        'labels': x['y']
    })
    
    time_end = time.time()
    print('time cost', time_end - time_start, 's')

    print(len(dataset))
    for data in DataLoader(dataset, batch_size=1, shuffle=True):
        print(data)
        break
