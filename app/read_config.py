'''This script opens config.yaml into a cfg variable so that I can import it without extra code'''

import yaml


with open('config.yaml', 'r', encoding='utf8') as file:
    cfg = yaml.safe_load(file)
