'''This script calculates mean and std for normalization'''

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


train_images_path = ''

# input format:
# /root:
#    /train_images:
#       /class1
#       /class2


transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
])

dataset = torchvision.datasets.ImageFolder(root=train_images_path, transform=transformations)
data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)


def get_mean_std(loader):

    channels_sum = 0
    channels_squared_sum = 0
    batches = 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        batches += 1

    mean = channels_sum / batches
    std = (channels_squared_sum / batches - mean**2)**0.5

    return mean, std


mean, std = get_mean_std(data_loader)
print(f'mean: {mean}, std: {std}')
