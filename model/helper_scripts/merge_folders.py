'''Helps to merge smiling and unsmiling folders after split_data.py'''

import shutil
import os

current_folder = os.getcwd()

folders = ['folder1', 'folder2']

content_list = {}

for index, val in enumerate(folders):

	path = os.path.join(current_folder, val)
	content_list[folders[index]] = os.listdir(path)


merge_folder = 'output_folder'

merge_folder_path = os.path.join(current_folder, merge_folder)


for sub_dir in content_list:

	for contents in content_list[sub_dir]:

		path_to_content = sub_dir + "/" + contents

		dir_to_move = os.path.join(current_folder, path_to_content)
  
		shutil.move(dir_to_move, merge_folder_path)
