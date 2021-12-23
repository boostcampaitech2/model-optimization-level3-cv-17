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
from src.MnasNet import MnasNet
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.torch_utils import check_runtime, model_info
from src.utils.lr_scheculer import CosineAnnealingWarmUpRestarts
from src.decompositions import cp_decomposition_conv_layer, tucker_decomposition_conv_layer

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

def load_ckp(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

def prune(model, amount=0.3):
    import torch.nn.utils.prune as prune
    # Prune model to requested global sparsity
    print('Pruning model... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent
    print(' %.3g global sparsity' % sparsity(model))

def sparsity(model):
    # Return global model sparsity
    a, b = 0., 0.
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a

def model_equivalence(model_1, model_2, device, rtol=1e-05, atol=1e-08, num_tests=100, input_size=(1,3,32,32)):

    model_1.to(device)
    model_2.to(device)

    for _ in range(num_tests):
        x = torch.rand(size=input_size).to(device)
        y1 = model_1(x).detach().cpu().numpy()
        y2 = model_2(x).detach().cpu().numpy()
        if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
            print("Model equivalence test sample failed: ")
            print(y1)
            print(y2)
            return False

    return True

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

    model = MnasNet(n_class=6, input_size=192, width_mult=1.0)
    prune(model)

    # model_instance = Model(model_config, verbose=True) #model load
    # model_path = os.path.join(log_dir, "best.pt") #model 저장 경로 정의

    # -- pre-trained
    model_path = 'pretrained/MnasNet.pth'
    model_dict = torch.load(model_path, map_location=device)
    
    model_dict_keys = list(model_dict["state_dict"].keys())
    for key in model_dict_keys:
        if key in ['module.classifier.1.weight', 'module.classifier.1.bias']:
            model_dict["state_dict"].pop(key)
        else:
            model_dict["state_dict"][key[7:]] = model_dict["state_dict"].pop(key)
    # for key in ['classifier.weight', 'classifier.bias']:
    #     model_dict.pop(key)
    model.load_state_dict(model_dict['state_dict'], strict=False)
    model.to(device)

    # print(f"Model save path: {model_path}")
    # if os.path.isfile(model_path):
    #     model_instance.model.load_state_dict(
    #         torch.load(model_path, map_location=device)
    #     )
    # model_instance.model.to(device)

    # -- decompose
    # model.eval()
    # model.cpu()
    # N = len(model.features._modules.keys())

    # for i, key in enumerate(model.features._modules.keys()):
    #     # print('key', key)
    #     if i >= N - 2:
    #         break
    #     for sub_i, sub_key in enumerate(model.features._modules[key]._modules.keys()):
    #         # print('sub_key', model.features._modules[key]._modules[sub_key])
    #         if isinstance(model.features._modules[key]._modules[sub_key], torch.nn.modules.conv.Conv2d):
    #             conv_layer = model.features._modules[key]._modules[sub_key]
    #             if conv_layer.kernel_size[1] != 1:
    #                 # rank = max(conv_layer.weight.data.numpy().shape)//3
    #                 decomposed = tucker_decomposition_conv_layer(conv_layer)
    #                 model.features._modules[key]._modules[sub_key] = decomposed

    #         if isinstance(model.features._modules[key]._modules[sub_key], torch.nn.Sequential):
    #             for sub_i2, sub_key2 in enumerate(model.features._modules[key]._modules[sub_key]._modules.keys()):
    #                 if isinstance(model.features._modules[key]._modules[sub_key]._modules[sub_key2], torch.nn.modules.conv.Conv2d):
    #                     conv_layer = model.features._modules[key]._modules[sub_key]._modules[sub_key2]
    #                     if conv_layer.kernel_size[1] != 1:
    #                         # rank = max(conv_layer.weight.data.numpy().shape)//3
    #                         decomposed = tucker_decomposition_conv_layer(conv_layer)
    #                         model.features._modules[key]._modules[sub_key]._modules[sub_key2] = decomposed
    #     model.features._modules['1']._modules['0']._modules['0'] = nn.Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     model.features._modules['2']._modules['conv']._modules['3']._modules['0'] = nn.Conv2d(48, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     model.features._modules['3']._modules['conv']._modules['3']._modules['0'] = nn.Conv2d(36, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     model.features._modules['4']._modules['conv']._modules['3']._modules['0'] = nn.Conv2d(36, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     model.features._modules['5']._modules['conv']._modules['3']._modules['0'] = nn.Conv2d(36, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     model.features._modules['6']._modules['conv']._modules['3']._modules['0'] = nn.Conv2d(60, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     model.features._modules['7']._modules['conv']._modules['3']._modules['0'] = nn.Conv2d(60, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     model.features._modules['8']._modules['conv']._modules['3']._modules['0'] = nn.Conv2d(120, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     model.features._modules['9']._modules['conv']._modules['3']._modules['0'] = nn.Conv2d(240, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     model.features._modules['10']._modules['conv']._modules['3']._modules['0'] = nn.Conv2d(240, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     model.features._modules['11']._modules['conv']._modules['3']._modules['0'] = nn.Conv2d(240, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     model.features._modules['12']._modules['conv']._modules['3']._modules['0'] = nn.Conv2d(288, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     model.features._modules['13']._modules['conv']._modules['3']._modules['0'] = nn.Conv2d(288, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     model.features._modules['14']._modules['conv']._modules['3']._modules['0'] = nn.Conv2d(576, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     model.features._modules['15']._modules['conv']._modules['3']._modules['0'] = nn.Conv2d(576, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     # model.features._modules['16']._modules['conv']._modules['3']._modules['0'] = nn.Conv2d(576, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     # model.features._modules['17']._modules['conv']._modules['3']._modules['0'] = nn.Conv2d(576, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     torch.save(model, 'pretrained/decomposed.pt')

    # # -- decomposed model reload
    # model = torch.load('pretrained/decomposed.pt')
    # model.to(device)
    # print('composition done')
    # print(model)
    # assert 0 == 1
    # -- quantization
    # model.train()
    # model = torch.quantization.fuse_modules(model, [["conv1", "bn1", "relu"]], inplace=True)
    

    # assert model_equivalence(model_1=model, model_2=fused_model, device=cpu_device, rtol=1e-03, atol=1e-06, num_tests=100, input_size=(1, 3, 224, 224)), "Fused model is not equivalent to the original model!"
    
    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Create optimizer, scheduler, criterion
    optimizer = torch.optim.Adam(
        model.parameters(), lr=data_config["INIT_LR"], weight_decay=1e-6
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=0.000001)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer=optimizer,
    #     max_lr=data_config["INIT_LR"],
    #     steps_per_epoch=len(train_dl),
    #     epochs=data_config["EPOCHS"],
    #     pct_start=0.05,
    #     # div_factor=25,
    #     final_div_factor=10000.0, # default: 10000.0
    # )
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

    # resume
    # ckp_path =unfold '/opt/ml/code/exp/mobilenetv2_192_0.5/best.pt'
    # model, optimizer, _ = load_ckp(ckp_path, model, optimizer)

    # Create trainer
    trainer = TorchTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_path=os.path.join(log_dir, "best.pt"),
        verbose=1,
        wandb_name='MnasNet',
    )
    best_acc, best_f1, best_cnt = trainer.train(
        train_dataloader=train_dl,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=val_dl if val_dl else test_dl,
    )

    # evaluate model with test set
    model_path = os.path.join(log_dir, "best.pt")[:-3] + f'_{(best_cnt - 1) % 3}.ts'
    # model.load_state_dict(torch.load(model_path)['state_dict'])
    model = torch.jit.load(model_path)
    test_loss, test_f1, test_acc = trainer.test(
        model=model, test_dataloader=val_dl if val_dl else test_dl
    )

    return test_loss, test_f1, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model",
        default="configs/model/MnasNet.yaml",
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

