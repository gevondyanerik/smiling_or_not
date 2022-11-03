'''Training pipeline'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from data_preparation import train_loader, val_loader

from tqdm import tqdm
from itertools import chain
from read_config import cfg

import warnings
warnings.filterwarnings('ignore')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if __name__ == '__main__':
    print(f'Training on {device}')

model = resnet50(weights=ResNet50_Weights.DEFAULT)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 1)
model.to(device)

loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg['scheduler_patience'], threshold=cfg['scheduler_threshold'], verbose=True)


def get_round(number):
    '''Rounds predict with a 0 threshold'''

    return 0 if number < 0 else 1


def train_step(train_loader, model, optimizer, loss_function):
    '''Training loop'''

    model.train()

    epoch_loss = 0.
    correct = 0
    total = 0

    train_loop = tqdm(train_loader, leave=False)

    for images, labels in train_loop:

        # data to cuda(if able)
        images = images.to(device)
        labels = labels.unsqueeze(1)
        labels = labels.to(torch.float32)
        labels = labels.to(device)

        # forward
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_function(predictions, labels)

        # backward
        loss.backward()
        optimizer.step()

        # evaluation
        epoch_loss += loss.item()
        mean_loss = epoch_loss / len(train_loader)

        labels = list(chain(*labels.tolist()))
        predictions = list(chain(*predictions.tolist()))
        predictions = list(map(lambda predict: get_round(predict), predictions))

        correct += sum([predict == label for predict, label in zip(predictions, labels)])
        total += len(labels)

        train_loop.set_postfix(loss=mean_loss) 

    scheduler.step(mean_loss)

    accuracy = correct / total * 100
    print(f'TRAIN | ACCURACY: {accuracy} | LOSS: {mean_loss}')

    return (accuracy, mean_loss)


def val_step(val_loader, model, loss_function):
    '''Validation loop'''

    model.eval()

    epoch_loss = 0.
    correct = 0
    total = 0

    val_loop = tqdm(val_loader, leave=False)

    for images, labels in val_loop:

        # data to cuda(if able)
        images = images.to(device)
        labels = labels.unsqueeze(1)
        labels = labels.to(torch.float32)
        labels = labels.to(device)

        # predict
        predictions = model(images)

        # evaluation
        loss = loss_function(predictions, labels)

        epoch_loss += loss.item()
        mean_loss = epoch_loss / len(val_loader)

        labels = list(chain(*labels.tolist()))
        predictions = list(chain(*predictions.tolist()))
        predictions = list(map(lambda predict: get_round(predict), predictions))

        correct += sum([predict == label for predict, label in zip(predictions, labels)])
        total += len(labels)

        val_loop.set_postfix(loss=mean_loss) 

    accuracy = correct / total * 100
    print(f'VAL   | ACCURACY: {accuracy} | LOSS: {mean_loss}')

    return (accuracy, mean_loss)


class EarlyStopping():
    '''Returns True if no improvement after a given number of epochs in a row.

       counter[int]: current number of no improvement epochs,
       patience[int]: max number of no improvement epochs,
       delta[float]: loss minus previous_loss less than delta means 'no imrovement',
       previous_loss[float]: loss of a previous epoch.'''

    def __init__(self, counter=0, patience=cfg['es_patience'], delta=cfg['es_delta'], previous_loss=1.):
        self.counter = counter
        self.patience = patience
        self.delta = delta
        self.previous_loss = previous_loss

    def __call__(self, loss):

        if self.previous_loss - loss <= self.delta:
            self.counter += 1

            if self.counter >= self.patience:
                print('\nEarly stop...')
                return True

        else:
            self.counter = 0

        self.previous_loss = loss
        print(f'Current early stop counter: {self.counter}/{self.patience}')


early_stopping = EarlyStopping()


def save_checkpoint(state, epoch, val_accuracy, val_loss, folder=cfg['checkpoints_folder']):
    '''Saves model into cfg checkpoints folder'''

    print('\nSaving model...')
    torch.save(state, f'{folder}\checkpoint_ep-{epoch + 1}_acc-{str(val_accuracy)[:5]}_loss-{str(val_loss)[:5]}.pth.tar')


start_epoch = 0     # allows to track how many epochs a model has trained when training saved checkpoints
best_accuracy = 0

train_acc_per_epoch = []
val_acc_per_epoch = []

train_loss_per_epoch = []
val_loss_per_epoch = []


if cfg['load_checkpoint']:     # if cfg load_checkpoint is True, the model from cfg load_checkpoint_path will be loaded

    print('\nLoading checkpoint...\n')
    checkpoint = torch.load(cfg['load_checkpoint_path'], map_location=device)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['val_accuracy']

    train_acc_per_epoch = checkpoint['train_acc_per_epoch']
    val_acc_per_epoch = checkpoint['val_acc_per_epoch']

    train_loss_per_epoch = checkpoint['train_loss_per_epoch']
    val_loss_per_epoch = checkpoint['val_loss_per_epoch']


if __name__ == "__main__":

    '''Runs train_loop and val_loop in turn, appending accuracy and loss into the lists above.
    If cfg save_checkpoints is True, each model that has more accuracy
    than all of previous, will be saved into the cfg checkpoints_folder'''

    for epoch in range(start_epoch, cfg['epochs'] + start_epoch):

        print(f'\nEPOCH |{epoch + 1}/{cfg["epochs"] + start_epoch}|')

        train_accuracy, train_loss = train_step(train_loader, model, optimizer, loss_function)
        val_accuracy, val_loss = val_step(val_loader, model, loss_function)

        train_acc_per_epoch.append(train_accuracy)
        val_acc_per_epoch.append(val_accuracy)

        train_loss_per_epoch.append(train_loss)
        val_loss_per_epoch.append(val_loss)

        if cfg['save_checkpoints'] and val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            tmp_dict = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'train_acc_per_epoch': train_acc_per_epoch,
                'val_acc_per_epoch': val_acc_per_epoch,
                'train_loss_per_epoch': train_loss_per_epoch,
                'val_loss_per_epoch': val_loss_per_epoch,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }

            save_checkpoint(tmp_dict, epoch, val_accuracy, val_loss)

        if early_stopping(val_loss):
            break
