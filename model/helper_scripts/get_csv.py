'''Helps to get train/val/test_csv.'''

import os
import pandas as pd

images_path = ''

images = os.listdir(images_path)
labels = []

for image in images:

  if image.startswith('smiling'):
    labels.append(1)

  elif image.startswith('unsmiling'):
    labels.append(0)

  else:
    print('something wrong...')
    break

tmp_dict = {'image': images,
            'label': labels}

data = pd.DataFrame(tmp_dict, index=None)
data.to_csv('train_csv.csv', index=None)