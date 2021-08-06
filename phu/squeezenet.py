#! /usr/bin/env python

from torchvision import models
from torch.nn import Conv2d
from torch.optim import SGD
from train import train

model = models.squeezenet1_1(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

old_classifier = model.classifier[1]
model.classifier[1] = Conv2d(old_classifier.in_channels, 67, kernel_size=old_classifier.kernel_size)

learning_rate = 2e-5
weight_decay = learning_rate * 1e-6
optimizer = SGD(model.classifier[1].parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
train('squeezenet', model, optimizer)
