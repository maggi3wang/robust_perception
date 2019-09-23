"""
Simple script to separate simulated images into a testing and training set,
as well as move all label images and pose text files into a separate folder.

Run this whenever new data is generated.
"""

import os
import shutil

NUM_CLASSES = 5
NUM_TRAINING = 1000

classification = 'images/classification'
classification_clean_dir = 'images/classification_clean'
training_set_dir = classification_clean_dir + '/training_set'
testing_set_dir = classification_clean_dir + '/testing_set'
labels_and_poses_dir = 'images/labels_and_poses'

dirs_with_subdirs = [training_set_dir, testing_set_dir, labels_and_poses_dir]

try:
	shutil.rmtree(classification_clean_dir, ignore_errors=False, onerror=None)
	shutil.rmtree(labels_and_poses_dir, ignore_errors=False, onerror=None)
except:
	print("whoops, couldn't remove the existing dirs")

os.mkdir(classification_clean_dir)
os.mkdir(training_set_dir)
os.mkdir(testing_set_dir)
os.mkdir(labels_and_poses_dir)

for main_dir in dirs_with_subdirs:
	for i in range(1, NUM_CLASSES + 1):
		os.mkdir("{}/{}".format(main_dir, i))

# Now copy over files from the classification folder into the right directors

for num_dir in os.listdir(classification):		# 1 thru 5
	for i, file in enumerate(os.listdir("{}/{}".format(classification, num_dir))):
		filename = os.fsdecode(file)
		split_filename = filename.split("_")
		num = int(split_filename[1])

		set_dir = ""
		if num < NUM_TRAINING:
			set_dir = training_set_dir
		else:
			set_dir = testing_set_dir

		original_path = "{}/{}/{}".format(classification, num_dir, filename)

		if not "color" in filename:
			set_dir = labels_and_poses_dir

		new_path = "{}/{}/{}".format(set_dir, num_dir, filename)

		shutil.copyfile(original_path, new_path)

print("DONE :)")
