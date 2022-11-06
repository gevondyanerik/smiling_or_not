'''This script takes only one image and returns prediction(smiling/unsmiling)'''

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

model_statement = torch.load('model_statement.pth.tar', map_location=device)
model.load_state_dict(model_statement['state_dict'])

transformations = transforms.Compose([
  transforms.Resize((256, 256)),
  transforms.ToTensor(),
  transforms.Normalize(torch.Tensor([0.5079, 0.4671, 0.4429]), torch.Tensor([0.2924, 0.2688, 0.2716])),
])


def get_predict(path=None, image=None):
    '''Takes path(for local inference) or PIL image(for app.py) and returns prediction(smiling/unsmiling).'''

    if path:
        image = Image.open(path)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = transformations(image).float()
    image = image.unsqueeze(0) 

    image.to(device)

    prediction = model(image)
    prediction = list(chain(*prediction.tolist()))[0]

    return 'smiling' if prediction > 0 else 'unsmiling'     # prediction threshold is 0


if __name__ == '__main__':
    print(get_predict(path=input('Enter path to a image: ')))
