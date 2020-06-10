import numpy as np
import os
import glob
from skimage import io
import matplotlib.pyplot as plt
import csv

from helper_function import print_yellow, print_green, print_red

docker = True
if docker:
	dataset_dir = '../data/Data_registered/'
else:
	dataset_dir = './data/Phase_fluo_Cells/Data_registered/'

dataset = 'train'
csv_file = dataset_dir+'{}.csv'.format(dataset)
hrSLM_file = dataset_dir+'{}_hrSLIM.csv'.format(dataset)

img_fnames =[]; map_fnames = []
with open(csv_file, newline='') as f:
	csv_reader = csv.reader(f)
	for row in csv_reader:
# 		print(row)
		img_fnames.append(row[0]); map_fnames.append(row[1])

# print(len(img_fnames)); print_green(len(map_fnames))
img_fnames1= img_fnames; map_fnames1 = map_fnames

img_fnames =[]; map_fnames = []
with open(hrSLM_file, newline='') as f:
	csv_reader = csv.reader(f)
	for row in csv_reader:
# 		print(row)
		img_fnames.append(row[0]); map_fnames.append(row[1])

# rnd_indices = np.random.randint(0,len(img_fnames),4)
# for indx in rnd_indices.tolist():
# 	print(img_fnames1[indx]); print_green(img_fnames[indx])
# 	print(map_fnames1[indx]); print_red(map_fnames[indx])

# map_list =[]
# for indx in range(len(img_fnames)):
# 	mSLM_file = img_fnames1[indx]; hrSLM_file = img_fnames[indx]; map_file = map_fnames[indx]
# 	compl= os.path.exists(dataset_dir+mSLM_file)\
# 	 and os.path.exists(dataset_dir+hrSLM_file) and os.path.exists(dataset_dir+map_file)
# 	if compl:
# 		img_idx = indx
# 		break

plt.ion();fig = plt.figure()
for img_idx in range(len(img_fnames)):
	plt.clf()
	mSLM_file = img_fnames1[img_idx]; hrSLM_file = img_fnames[img_idx]; map_file = map_fnames[img_idx]
	mSLM_img = io.imread(dataset_dir+mSLM_file)
	hrSLM_img = io.imread(dataset_dir+hrSLM_file)
	map = io.imread(dataset_dir+map_file)
	ax = fig.add_subplot(1,3,1); cax=ax.imshow(mSLM_img, cmap ='Blues'); fig.colorbar(cax); ax.set_title('mSLM')
	ax = fig.add_subplot(1,3,2); cax=ax.imshow(hrSLM_img, cmap ='Blues'); fig.colorbar(cax); ax.set_title('hrSLM')
	ax = fig.add_subplot(1,3,3); cax=ax.imshow(map, cmap ='Blues'); fig.colorbar(cax); ax.set_title('Map'); ax.set_xlabel('{}'.format(np.unique(map)))
	plt.tight_layout(); plt.show(); plt.pause(1)
