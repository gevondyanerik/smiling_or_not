'''This script helps to rename parsed jpg files'''

import os

folder = r''
images = os.listdir(folder)

for index, image in enumerate(images):
    os.rename(os.path.join(folder, image), os.path.join(f'{folder}/smiling{index}.jpg'))
