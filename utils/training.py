from .base_classes import *
from .common import save_model
from .eveluation import test
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch

def train(
    model: torch.nn.Module,
    train_config: TrainConfigBase,
    scheduler: TrainSchedulerBase,
):
    model.train()
    optimizer = train_config.create_optimizer(model)
    dataloader = DataLoader(train_config.ds_train, batch_size=train_config.batch_size, collate_fn=lambda b:model.collate_fn(b))
    e_steps = len(dataloader)
    scheduler.on_start(epoch_steps=e_steps)
    eval_by_steps_1 = train_config.eval_by_steps - 1

    def get_lr():
        return train_config.optimizer.param_groups[0]['lr']

    acum_loss = 0
    for epoch in range(0, train_config.num_epoches):
        print(f"epoch={epoch+1}/{train_config.num_epoches} lr={get_lr():>5.2e}")
        for step, (x, y) in enumerate(tqdm(dataloader)):
            global_step = epoch * e_steps + step
            
            optimizer.zero_grad()
            y_pred = model(x)
            loss = train_config.loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            acum_loss += loss.item()
            t_loss = acum_loss / (global_step + 1)
            scheduler.on_step_end(epoch, global_step, t_loss)

            # evaluate the model
            if step % train_config.eval_by_steps == eval_by_steps_1 or step == e_steps - 1:
                print(f"\nepoch={epoch+1}/{train_config.num_epoches} step={step+1} train_loss={t_loss:>5.2} lr={get_lr():>5.2e}")
                # v_score = test(model, train_config, is_eval=True)
                # model.train()
                # if epoch + 1 >= train_config.start_saving_epoch and v_score > max_score:
                #     max_score = v_score
                #     save_model(model, train_config.get_model_save_path())

                # print(f"\nepoch={epoch+1}/{train_config.num_epoches} step={step+1} train_loss={t_loss:>5.2} v_score={v_score} lr={get_lr():>5.2e}")



