from utils import *
import os
from summarizer.BertPGN4Car import *
import platform

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    # parse args
    args = arg_parser.parse_args()
    set_proxy(args)
    
    args = BertPGN4CarArguments_A100()
    summarizer = BertPGN4CarSummizer()
    summarizer.train(args)

    # 执行关机命令
    if platform.system() == "Linux":
        try:
            os.system("shutdown -h now")
            print("正在关机...")
        except Exception as e:
            print(f"关机失败: {e}")
    elif platform.system() == "Windows":
        try:
            os.system("shutdown /s /t 0")
            print("正在关机...")
        except Exception as e:
            print(f"关机失败: {e}")
