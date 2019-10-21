"""
Rename files by adding a constant to the number
(1_0000_color.png to 1_2075_color.png, for all files in the directory)
and move images from one folder to another.
"""

import os
import shutil

package_dir = os.path.dirname(os.path.abspath(__file__))
print(package_dir)

initial_dir = os.path.join(package_dir, 'images1/classification/5')
final_dir = os.path.join(package_dir, 'images/classification/5')

CONSTANT = 2070 + 5

# for num_dir in os.listdir(classification):		# 1 thru 5
for i, file in enumerate(os.listdir(initial_dir)):
	filename = os.fsdecode(file)
	split_filename = filename.split("_")
	split_filename[1] = str(int(split_filename[1]) + CONSTANT)

	original_path = "{}/{}".format(initial_dir, filename)
	new_filename = '_'.join(split_filename)
	new_path = "{}/{}".format(final_dir, new_filename)

	# print('original_path: {}, new_path: {}'.format(original_path, new_path))
	shutil.copyfile(original_path, new_path)

print("DONE :)")