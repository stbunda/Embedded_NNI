# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from __future__ import annotations

import sys, os
sys.path.append(os.getcwd())
import time
from typing import Callable

import torch
import torch.nn.functional as F
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageNet, MNIST

import nni
from nni.contrib.compression.quantization import QATQuantizer
from nni.contrib.compression.utils import TorchEvaluator
from nni.common.types import SCHEDULER
from torchvision.models.mobilenetv2 import mobilenet_v2


torch.manual_seed(1024)
device = 'cuda:0'

ImageNet_path = 'data/ImageNet' #'/deepstore/datasets/dmb/MachineLearning/ImageNet/ILSVRC/Data/CLS-LOC/'

#train_loader = torch.utils.data.DataLoader(ImageNet(ImageNet_path, split='train'),
#                                               batch_size=64,
#                                               shuffle=True)
val_set = ImageNet(ImageNet_path, split='val', transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
test_loader = torch.utils.data.DataLoader(val_set,
                                          batch_size=64,
                                          shuffle=False)
                                               


#def training_step(batch, model):
#    x, y = batch[0].to(device), batch[1].to(device)
#    logits = model(x)
#    loss: torch.Tensor = F.nll_loss(logits, y)
#    return loss
#
#
#def training_model(model: torch.nn.Module, optimizer: Optimizer, training_step: Callable, scheduler: SCHEDULER | None = None,
#                   max_steps: int | None = None, max_epochs: int | None = None):
#    model.train()
#    max_epochs = max_epochs if max_epochs else 1 if max_steps is None else 100
#    current_steps = 0
#
#    # training
#    for epoch in range(max_epochs):
#        print(f'Epoch {epoch} start!')
#        for batch in train_loader:
#            optimizer.zero_grad()
#            loss = training_step(batch, model)
#            loss.backward()
#            optimizer.step()
#            current_steps += 1
#            if max_steps and current_steps == max_steps:
#                return
#        if scheduler is not None:
#            scheduler.step()


def evaluating_model(model: torch.nn.Module):
    model.eval()
    # testing
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += preds.eq(y.view_as(preds)).sum().item()
    return correct / len(val_set)


model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True).to(device)
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

start = time.time()
# training_model(model, optimizer, training_step, None, None, 5)
print(f'pure training 5 epochs: {time.time() - start}s')
start = time.time()
acc = evaluating_model(model)
print(f'pure evaluating: {time.time() - start}s    Acc.: {acc}')

#optimizer = nni.trace(SGD)(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
#evaluator = TorchEvaluator(training_model, optimizer, training_step)  # type: ignore
#
#config_list = [{
#    'op_types': ['Conv2d'],
#    'target_names': ['_input_', 'weight', '_output_'],
#    'quant_dtype': 'int8',
#    'quant_scheme': 'affine',
#    'granularity': 'default',
#},{
#    'op_types': ['ReLU6'],
#    'target_names': ['_output_'],
#    'quant_dtype': 'int8',
#    'quant_scheme': 'affine',
#    'granularity': 'default',
#}]
#
#quantizer = QATQuantizer(model, config_list, evaluator, len(train_loader))
#real_input = next(iter(train_loader))[0].to(device)
#quantizer.track_forward(real_input)
#
#start = time.time()
#_, calibration_config = quantizer.compress(None, max_epochs=5)
#print(f'pure training 5 epochs: {time.time() - start}s')
#
#print(calibration_config)
#start = time.time()
#acc = evaluating_model(model)
#print(f'quant evaluating: {time.time() - start}s    Acc.: {acc}')
