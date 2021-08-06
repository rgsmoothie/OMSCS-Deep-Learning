#! /usr/bin/env python

from torchvision import models
from torch.nn import Linear
from torch.optim import SGD
from train import train

model = models.mobilenet_v3_small(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.classifier[3] = Linear(model.classifier[3].in_features, 67)

learning_rate = 0.5
weight_decay = learning_rate * 1e-6
optimizer = SGD(model.classifier.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
train('mobilenet', model, optimizer)
