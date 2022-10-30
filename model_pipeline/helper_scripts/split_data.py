'''This scripts splits the data to training, validation and testing'''

# pip install split-folders

import splitfolders

# input format:
# \root:
#    \data:
#       \class1
#       \class2

splitfolders.ratio('\dataset\data', output='\dataset', seed=999, ratio=(.7, .2, .1), group_prefix=None, move=False)
