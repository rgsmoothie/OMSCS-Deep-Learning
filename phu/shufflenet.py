#! /usr/bin/env python

from torchvision import models
from torch.nn import Linear
from torch.optim import SGD
from train import train

model = models.shufflenet_v2_x0_5(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = Linear(model.fc.in_features, 67)

learning_rate = 38.4
weight_decay = learning_rate * 1e-6
optimizer = SGD(model.fc.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
train('shufflenet', model, optimizer)
