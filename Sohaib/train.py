from os.path import basename, dirname
import glob
import math

from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import io
from torchvision.transforms import Normalize
import torch
from tqdm.auto import tqdm
import time

class Fruits360Dataset(Dataset):
    def __init__(self, directory, transform=None, target_transform=None, device='cpu'):
        super(Dataset, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    
#     def cuda():
        


def train_model(name, model, optimizer, scheduler, dataloader, criterion):
    batch_count = len(dataloader)
    count_total = 0
    
    model.train()
    loss_total = 0
    accuracy_total = 0

    for batch, (inputs, labels) in enumerate(dataloader, 1):
        inputs = inputs.cuda()
        labels = labels.cuda()
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
#         print(f'Batch {batch}/{batch_count} accuracy: {accuracy:.4f}, loss: {loss:.4f}')

        loss.backward()
        optimizer.step()

    accuracy = accuracy_total / count_total
    loss = loss_total / count_total
#     print(f'Train accuracy: {accuracy:.4f}, loss: {loss:.4f}')

    scheduler.step(loss)
    torch.save(model, f'{name}.pt')
    return accuracy, loss

def evaluate_model(model, dataloader, criterion):
    batch_count = len(dataloader)
    count_total = 0

    model.eval()
    loss_total = 0
    accuracy_total = 0

    with torch.set_grad_enabled(False):
        for batch, (inputs, labels) in enumerate(dataloader, 1):
            inputs = inputs.cuda()
            labels = labels.cuda()
            batch_size = inputs.shape[0]
            count_total += batch_size

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            accuracy = preds.eq(labels).sum()
            accuracy_total += accuracy
            accuracy = accuracy / batch_size

            loss = criterion(outputs, labels)
            loss_total += loss * batch_size
#             print(f'Batch {batch}/{batch_count} accuracy: {accuracy:.4f}, loss: {loss:.4f}')

    accuracy = accuracy_total / count_total
    loss = loss_total / count_total
#     print(f'Test accuracy: {accuracy:.4f}, loss: {loss:.4f}')
    return accuracy, loss

def train(name, model, optimizer, epochs=10, scheduler = None, train_dataloader=None, test_dataloader=None):
    print("Current Device:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print(next(model.parameters()).is_cuda)
    model.to('cuda')
    print(next(model.parameters()).is_cuda)
    device = torch.device('cuda:0')
    
    criterion = CrossEntropyLoss().cuda()
    if scheduler is None:
        scheduler = ReduceLROnPlateau(
            optimizer,
            'min',
            threshold=0.05,
            factor=0.5,
            patience=0,
            verbose=True)
#     transform = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).cuda()
    
#     train_dataset = Fruits360Dataset('fruits-360/Training', transform)
#     test_dataset = Fruits360Dataset('fruits-360/Test', transform)
    if train_dataloader is None or test_dataloader is None:
        transform = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).cuda()
        train_dataset = Fruits360Dataset('fruits-360/Training', transform)
        test_dataset = Fruits360Dataset('fruits-360/Test', transform)
        batch_size = math.ceil(len(train_dataset) / 10)
        train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True)

        test_dataloader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True)
    results = {
        'name':name,
        'epochs':epochs,
        'train_losses':[],
        'train_accs':[],
        'test_losses':[],
        'test_accs':[],
    }
    start = time.time()
    for epoch in tqdm(range(1, epochs + 1), desc='Epoch'):
#         print(f'Epoch {epoch}/{epochs}')
        train_acc, train_loss = train_model(name, model, optimizer, scheduler, train_dataloader, criterion)
        test_acc, test_loss = evaluate_model(model, test_dataloader, criterion)
        results['train_accs'].append(train_acc.detach().cpu().numpy().item())
        results['train_losses'].append(train_loss.detach().cpu().numpy().item())
        results['test_accs'].append(test_acc.detach().cpu().numpy().item())
        results['test_losses'].append(test_loss.detach().cpu().numpy().item())
        print(f'Epoch: {epoch} Train: {train_acc:.4f}, {train_loss:.4f} Test: {test_acc:.4f}, {test_loss:.4f}', end='\r')
    print('')
    end = time.time()
    results['time'] = end-start
    print(f'Time Taken: {end-start}')
    return results