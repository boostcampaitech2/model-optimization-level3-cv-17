"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
import os
from datetime import datetime
from typing import Any, Dict, Tuple, Union

import random
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model
from src.mobilenetv3 import mobilenetv3_large
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.torch_utils import check_runtime, model_info
from src.decompositions import cp_decomposition_conv_layer

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

def train(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    log_dir: str,
    fp16: bool,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train."""
    # save model_config, data_config
    with open(os.path.join(log_dir, "data.yml"), "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, "model.yml"), "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)

    seed_everything(1995)

    net_large = mobilenetv3_large(model_config)
    # model_instance = Model(model_config, verbose=True) #model load
    # model_path = os.path.join(log_dir, "best.pt") #model 저장 경로 정의
    model_path = 'pretrained/mobilenetv3-large-1cd25616.pth'
    net_large.load_state_dict(torch.load(model_path))
    net_large.to(device)
    # print(f"Model save path: {model_path}")
    # if os.path.isfile(model_path):
    #     model_instance.model.load_state_dict(
    #         torch.load(model_path, map_location=device)
    #     )
    # model_instance.model.to(device)

    # -- decompose
    net_large.eval()
    net_large.cpu()
    N = len(net_large.features._modules.keys())
    for i, key in enumerate(net_large.features._modules.keys()):
        if i >= N - 2:
            break
        if isinstance(net_large.features._modules[key], torch.nn.modules.conv.Conv2d):
            conv_layer = net_large.features._modules[key]
            rank = max(conv_layer.weight.data.numpy().shape)//3
            decomposed = cp_decomposition_conv_layer(conv_layer, rank)
            net_large.features._modules[key] = decomposed
        torch.save(net_large, 'pretrained/decomposed.pt')

    # -- decomposed model reload
    net_large = torch.load('pretrained/decomposed.pt', map_location=device)
    net_large.to(device)

    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Create optimizer, scheduler, criterion
    optimizer = torch.optim.SGD(
        net_large.parameters(), lr=data_config["INIT_LR"], momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=data_config["INIT_LR"],
        steps_per_epoch=len(train_dl),
        epochs=data_config["EPOCHS"],
        pct_start=0.1,
    )
    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_config["DATA_PATH"])
        if data_config["DATASET"] == "TACO"
        else None,
        device=device,
    )

    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )

    # Create trainer
    trainer = TorchTrainer(
        model=net_large,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_path=os.path.join(log_dir, "best.pt"),
        verbose=1,
        wandb_name='mobilenetv3_large'
    )
    best_acc, best_f1 = trainer.train(
        train_dataloader=train_dl,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=val_dl if val_dl else test_dl,
    )

    # evaluate model with test set
    net_large.load_state_dict(torch.load(model_path))
    test_loss, test_f1, test_acc = trainer.test(
        model=net_large, test_dataloader=val_dl if val_dl else test_dl
    )

    return test_loss, test_f1, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model",
        default="configs/model/mobilenetv3_large.yaml",
        type=str,
        help="model config",
    )
    parser.add_argument(
        "--data", default="configs/data/taco.yaml", type=str, help="data config"
    )
    args = parser.parse_args()

    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)

    data_config["DATA_PATH"] = os.environ.get("SM_CHANNEL_TRAIN", data_config["DATA_PATH"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp", 'latest'))

    if os.path.exists(log_dir): 
        modified = datetime.fromtimestamp(os.path.getmtime(log_dir + '/best.pt'))
        new_log_dir = os.path.dirname(log_dir) + '/' + modified.strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(log_dir, new_log_dir)

    os.makedirs(log_dir, exist_ok=True)

    test_loss, test_f1, test_acc = train(
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
    )

