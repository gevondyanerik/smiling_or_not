'''This script takes only one image and returns prediction(Smiling/Unsmiling)'''

from read_config import cfg
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
from itertools import chain


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = resnet50(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 1)
model.to(device)

model_statement = torch.load(cfg['model_statement'], map_location=device)
model.load_state_dict(model_statement['state_dict'])

transformations = transforms.Compose([
  transforms.Resize((cfg['image_size'], cfg['image_size'])),
  transforms.ToTensor(),
  transforms.Normalize(torch.Tensor(cfg['mean']), torch.Tensor(cfg['std'])),
])


def get_predict(image):

    image = Image.open(image)
    image = transformations(image).float()
    image = image.unsqueeze(0) 
    image.to(device)

    prediction = model(image)
    prediction = list(chain(*prediction.tolist()))[0]

    return 'Smiling' if prediction else 'Unsmiling'     # return 'Smiling' if prediction > 0, otherwise 'Unsmiling'


if __name__ == '__main__':
    print(get_predict(input('Enter path to a image: ')))