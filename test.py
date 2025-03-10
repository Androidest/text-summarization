from utils import *
from datasets import *
from importlib import import_module
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    # parse args
    args = arg_parser.parse_args()
    print(f"========= Importing model: {args.model} ===========")

    # import models dynamically
    module = import_module(f'models.{args.model}')
    TrainConfig = module.TrainConfig
    Model = module.Model
    TrainScheduler = module.TrainScheduler

    train_config = TrainConfig()
    path = train_config.get_model_save_path(args.acc)
    print(f"Loading weights from: {path} ...")
    model = Model(train_config).to(train_config.device)
    model = load_model(model, path)
    scheduler = TrainScheduler(train_config, model)

    print(f"=================== Test =======================")
    assert os.path.exists(train_config.data_path_test)
    test_loss, score = test(model, train_config, return_all=True, verbose=True)
    print("Test result:")
    print(f"test_loss={test_loss:>5.2} score={score}")
