from os.path import basename, dirname
import glob
import math

from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import io
from torchvision.transforms import Normalize
import torch


class Fruits360Dataset(Dataset):
    def __init__(self, directory, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        paths = glob.glob(f'{directory}/**/*.jpg')
        self.images = [self.load_image(p) for p in paths]

        labels = [self.get_label(p) for p in paths]
        self.classes = set(labels)
        indices = {l: i for i, l in enumerate(self.classes)}
        self.indices = [indices[l] for l in labels]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.indices[index]

    def get_label(self, path):
        label = basename(dirname(path)).split(' ')[0]
        if self.target_transform:
            label = self.target_transform(label)
        return label

    def load_image(self, path):
        image = io.read_image(path).float()
        if self.transform:
            image = self.transform(image)
        return image


def train_model(name, model, optimizer, scheduler, dataloader, criterion):
    batch_count = len(dataloader)
    count_total = 0

    model.train()
    loss_total = 0
    accuracy_total = 0

    for batch, (inputs, labels) in enumerate(dataloader, 1):
        batch_size = inputs.shape[0]
        count_total += batch_size

        optimizer.zero_grad()
        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)
        accuracy = preds.eq(labels).sum()
        accuracy_total += accuracy
        accuracy = accuracy / batch_size

        loss = criterion(outputs, labels)
        loss_total += loss * batch_size
        print(f'Batch {batch}/{batch_count} accuracy: {accuracy:.4f}, loss: {loss:.4f}')

        loss.backward()
        optimizer.step()

    accuracy = accuracy_total / count_total
    loss = loss_total / count_total
    print(f'Train accuracy: {accuracy:.4f}, loss: {loss:.4f}')

    scheduler.step(loss)
    torch.save(model, f'{name}.pt')


def evaluate_model(model, dataloader, criterion):
    batch_count = len(dataloader)
    count_total = 0

    model.eval()
    loss_total = 0
    accuracy_total = 0

    with torch.set_grad_enabled(False):
        for batch, (inputs, labels) in enumerate(dataloader, 1):
            batch_size = inputs.shape[0]
            count_total += batch_size

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            accuracy = preds.eq(labels).sum()
            accuracy_total += accuracy
            accuracy = accuracy / batch_size

            loss = criterion(outputs, labels)
            loss_total += loss * batch_size
            print(f'Batch {batch}/{batch_count} accuracy: {accuracy:.4f}, loss: {loss:.4f}')

    accuracy = accuracy_total / count_total
    loss = loss_total / count_total
    print(f'Test accuracy: {accuracy:.4f}, loss: {loss:.4f}')


def train(name, model, optimizer):
    criterion = CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(
        optimizer,
        'min',
        threshold=0.05,
        factor=0.5,
        patience=0,
        verbose=True)

    transform = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_dataset = Fruits360Dataset('fruits-360/Training', transform)
    test_dataset = Fruits360Dataset('fruits-360/Test', transform)

    batch_size = math.ceil(len(train_dataset) / 10)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True)

    epochs = 100
    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}')
        train_model(name, model, optimizer, scheduler, train_dataloader, criterion)
        evaluate_model(model, test_dataloader, criterion)
