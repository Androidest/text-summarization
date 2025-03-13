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
    trainer = module.create_trainer()

    print(f"=================== Start training =======================")
    trainer.train()

    # print(f"=================== Test best model ======================")
    # model = load_model(model, train_config.get_model_save_path())
    # score = test(model, train_config, return_all=True, verbose=True)
    # save_model(model, train_config.get_model_save_path(score))
    # print("Test result:")
    # print(f"test_loss={test_lossscore={score}")
