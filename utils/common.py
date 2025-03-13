import os
import torch
import random
import numpy as np
import argparse

# common argument parser for entry scripts
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--model', type=str, help='choose a model', default='t5pgn') # default is the bert_opt model
arg_parser.add_argument('--suffix', type=str, help='model save file suffix', default='')
arg_parser.add_argument('--proxy', type=str, help='use proxy', const='http://127.0.0.1:1081', default=None, nargs='?')

def set_proxy(args):
    if args.proxy is None:
        return
    
    proxy_url = args.proxy
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    print(f"Using proxy on: {proxy_url}")

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True