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

    print(f"=================== Start training =======================")
    train_config = TrainConfig()
    assert os.path.exists(train_config.data_path_train)
    assert os.path.exists(train_config.data_path_val)
    assert os.path.exists(train_config.data_path_test)
    set_seed(train_config.random_seed)
    model = Model(train_config).to(train_config.device)
    
    scheduler = TrainScheduler(train_config, model)
    train(model, train_config, scheduler)

    # print(f"=================== Test best model ======================")
    # model = load_model(model, train_config.get_model_save_path())
    # score = test(model, train_config, return_all=True, verbose=True)
    # save_model(model, train_config.get_model_save_path(score))
    # print("Test result:")
    # print(f"test_loss={test_lossscore={score}")
