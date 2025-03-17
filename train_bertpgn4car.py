from utils import *
import os
from summarizer.BertPGN4Car import BertPointerGeneratorModel, BertPGN4CarArguments

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    # parse args
    args = arg_parser.parse_args()
    set_proxy(args)
    
    args = BertPGN4CarArguments()
    summarizer = BertPointerGeneratorModel()
    summarizer.train(args)
