from utils import *
from custom_datasets import *
from importlib import import_module
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    # parse args
    args = arg_parser.parse_args()
    set_proxy(args)
    
    print(f"========= Importing model: {args.model} ===========")

    # import models dynamically
    module = import_module(f'models.{args.model}')
