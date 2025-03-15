from utils import *
import os
from summarizer.T5PG4Car import T5PG4CarSummizer, T5PG4CarArguments

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    # parse args
    args = arg_parser.parse_args()
    set_proxy(args)
    
    args = T5PG4CarArguments()
    summarizer = T5PG4CarSummizer()
    summarizer.train(args)
