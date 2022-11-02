'''This script takes only one image and returns prediction(Smiling/Unsmiling)'''

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
  transforms.Normalize(torch.Tensor([0.5079, 0.4671, 0.4429] ), torch.Tensor([0.2924, 0.2688, 0.2716])),
])


def get_predict(image):
    '''Returns prediction(smiling/unsmiling).'''

    image = Image.open(image)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = transformations(image).float()
    image = image.unsqueeze(0) 
    
    image.to(device)

    prediction = model(image)
    prediction = list(chain(*prediction.tolist()))[0]

    return 'smiling' if prediction else 'unsmiling'


if __name__ == '__main__':
    print(get_predict(input('Enter path to a image: ')))