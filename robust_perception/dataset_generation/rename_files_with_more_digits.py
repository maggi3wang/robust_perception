"""
Make xxxx to xxxxx (add 0 to beginning) to make room for more data
"""

import os
import shutil

package_dir = os.path.dirname(os.path.abspath(__file__))
print(package_dir)

image_dir = os.path.join(package_dir, 'images/classification')

for num_dir in os.listdir(image_dir):		# 1 thru 5
	for i, file in enumerate(os.listdir("{}/{}".format(image_dir, num_dir))):
		filename = os.fsdecode(file)
		split_filename = filename.split("_")
		split_filename[1] = str(int(split_filename[1])).zfill(5)

		original_path = "{}/{}/{}".format(image_dir, num_dir, filename)
		new_filename = '_'.join(split_filename)
		new_path = "{}/{}/{}".format(image_dir, num_dir, new_filename)

		print('original_path: {}, new_path: {}'.format(original_path, new_path))
		shutil.move(original_path, new_path)

print("DONE :)")