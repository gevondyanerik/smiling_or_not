'''This file prepares data loaders for train.py and test.py'''

import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from sklearn.utils import shuffle
from read_config import cfg


train_transformations = transforms.Compose([
        transforms.Resize((cfg['image_size'], cfg['image_size'])),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
    ])


test_transformations = transforms.Compose([
        transforms.Resize((cfg['image_size'], cfg['image_size'])),
        transforms.ToTensor(),
    ])


class GetDataset(Dataset):
    '''Prepares dataset for torch DataLoader'''

    def __init__(self, data_csv, images_root, transformations):
        self.data_csv = shuffle(pd.read_csv(data_csv))
        self.root = images_root
        self.transformations = transformations

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, index):

        image = Image.open(os.path.join(self.root, self.data_csv.iloc[index, 0]))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = self.transformations(image)
        label = torch.tensor(int(self.data_csv.iloc[index, 1]))

        return (image, label)


train_dataset = GetDataset(cfg['train_csv'], cfg['train_images'], train_transformations)
val_dataset = GetDataset(cfg['val_csv'], cfg['val_images'], train_transformations)
test_dataset = GetDataset(cfg['test_csv'], cfg['test_images'], test_transformations)


train_loader = DataLoader(train_dataset, shuffle=True, batch_size=cfg['train_batch'])
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=cfg['train_batch'])
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=cfg['test_batch'])
