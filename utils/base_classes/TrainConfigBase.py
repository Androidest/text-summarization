import torch
import json
import os

class TrainConfigBase:
    # common parameters
    random_seed : int = 1
    pretrained_path : str = 'models_pretrained/xxx' # pretrained model path or Huggingface model name
    save_path : str = 'models_fine_tuned'
    model_name : str = 'xxx'
    num_epoches : int = 8
    start_saving_epoch : int = 1 # Save the model from the first epoch and count from 1
    batch_size : int = 128 # training batch_size
    eval_batch_size : int = 128 # evel batch_size
    test_batch_size : int = 1024 # test batch_size
    eval_by_steps : int = 200 # evaluate the model every 'eval_by_steps' steps when training
    optimizer = None # this will be set by the model's create_optimizer function
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    def create_optimizer(self, model: torch.nn.Module):
        raise NotImplementedError

    def loss_fn(self, logits, labels):
        return torch.nn.CrossEntropyLoss()(logits, labels)

    def get_model_save_path(self, suffix = None):
        path = f"{self.save_path}/{self.model_name}/{self.model_name}.pth"
        if suffix is not None:
            path = f"{path}.{suffix}"
        return path

    # def get_config_save_path(self, acc = None):
    #     path = f"{self.save_path}/{self.model_name}/{self.model_name}.config"
    #     return self.save_path_acc(path, acc)

    # def save(self, path: str):
    #     folder = os.path.dirname(path)
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)

    #     dic = {}
    #     bases = (str, float, int, bool, list, dict, tuple, set)
    #     keys = list(self.__annotations__.keys()) + list(self.__dict__.keys())
    #     for k in keys:
    #         v = self.__getattribute__(k)
    #         if (isinstance(v, bases)):
    #             dic[k] = v
    #     json_str = json.dumps(dic, indent=4)
    #     with open(path, 'w') as f:
    #         f.write(json_str)

    # def load(self, path: str):
    #     if not os.path.exists(path):
    #         raise FileNotFoundError(f"Model file not found: {path}")

    #     with open(path, 'r') as f:
    #         keys = list(self.__annotations__.keys()) + list(self.__dict__.keys())
    #         dic = json.loads(f.read())
    #         for k, v in dic.items():
    #             if k in keys:
    #                 self.__setattr__(k, v)
    #         return self