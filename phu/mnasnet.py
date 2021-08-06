#! /usr/bin/env python

from torchvision import models
from torch.nn import Linear
from torch.optim import SGD
from train import train

model = models.mnasnet0_5(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.classifier[1] = Linear(model.classifier[1].in_features, 67)

learning_rate = 2.6
weight_decay = learning_rate * 1e-3
optimizer = SGD(model.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
train('mnasnet', model, optimizer)
