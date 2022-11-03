'''Testing pipeline'''

import os
from tqdm import tqdm
from itertools import chain

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from read_config import cfg
from train import get_round, loss_function, optimizer
from data_preparation import test_loader


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Testing on {device}')

model = resnet50(weights=ResNet50_Weights.DEFAULT)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 1)
model.to(device)

train_accuracy = None
val_accuracy = None

train_loss = None
val_loss = None


def load_model(checkpoint_name, checkpoints_folder=cfg['checkpoints_folder']):
    '''Loads model statement into the resnet.'''

    print('\nLoading checkpoint...\n')
    checkpoint = torch.load(f'{checkpoints_folder}/{checkpoint_name}', map_location=device)

    global model
    global optimizer

    global train_accuracy
    global val_accuracy

    global train_loss
    global val_loss

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    train_accuracy = checkpoint['train_accuracy']
    val_accuracy = checkpoint['val_accuracy']

    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']


def test_step(test_loader, model, loss_function, checkpoint):
    model.eval()

    epoch_loss = 0.
    correct = 0
    total = 0

    test_loop = tqdm(test_loader, leave=False)

    for images, labels in test_loop:
        images = images.to(device)
        labels = labels.unsqueeze(1)
        labels = labels.to(torch.float32)
        labels = labels.to(device)

        predictions = model(images)

        loss = loss_function(predictions, labels)

        epoch_loss += loss.item()
        mean_loss = epoch_loss / len(test_loader)

        labels = list(chain(*labels.tolist()))
        predictions = list(chain(*predictions.tolist()))
        predictions = list(map(lambda predict: get_round(predict), predictions))

        correct += sum([predict == label for predict, label in zip(predictions, labels)])
        total += len(labels)

        test_loop.set_postfix(loss=mean_loss) 
    
    accuracy = correct / total * 100
    print(checkpoint[:-8])
    print(f'TEST | ACCURACY: {str(accuracy)[:5]} | LOSS: {str(mean_loss)[:5]}')


checkpoints = os.listdir(cfg['checkpoints_folder'])

for checkpoint in checkpoints:
  load_model(checkpoint)
  test_step(test_loader, model, loss_function, checkpoint)
