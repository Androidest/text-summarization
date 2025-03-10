from .base_classes import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import rouge

def test(
    model: torch.nn.Module, 
    train_config: TrainConfigBase, 
    verbose = False,
):
    model.eval()
    with torch.no_grad():
        b_size = train_config.test_batch_size
        dataloader = DataLoader(train_config.ds_test, batch_size=b_size, collate_fn=lambda b:model.collate_fn(b))
        
        if verbose:
            dataloader = tqdm(dataloader, desc="ROUGE Testing")

        preds = []
        labels = []
        for (x, y) in dataloader:
            y_pred = model.generate(x)
            preds.extend(y_pred)
            labels.extend(y)

        return rouge.Rouge().get_scores(hyps=preds, refs=labels, avg=True)