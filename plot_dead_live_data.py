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

# plt.ion();fig = plt.figure()
# for img_idx in range(len(img_fnames)):
# 	plt.clf()
# 	mSLM_file = img_fnames1[img_idx]; hrSLM_file = img_fnames[img_idx]; map_file = map_fnames[img_idx]
# 	mSLM_img = io.imread(dataset_dir+mSLM_file)
# 	hrSLM_img = io.imread(dataset_dir+hrSLM_file)
# 	map = io.imread(dataset_dir+map_file)
# 	ax = fig.add_subplot(1,3,1); cax=ax.imshow(mSLM_img, cmap ='Blues'); fig.colorbar(cax); ax.set_title('mSLM')
# 	ax = fig.add_subplot(1,3,2); cax=ax.imshow(hrSLM_img, cmap ='Blues'); fig.colorbar(cax); ax.set_title('hrSLM')
# 	ax = fig.add_subplot(1,3,3); cax=ax.imshow(map, cmap ='Blues'); fig.colorbar(cax); ax.set_title('Map'); ax.set_xlabel('{}'.format(np.unique(map)))
# 	plt.tight_layout(); plt.show(); plt.pause(1)

def check_one_pixel(xc, yc, xs, ys):
	x_list, y_list = [], []
	for i in range(-1,2):
		for j in range(-1,2):
			x, y = xc+i, yc +j
			if x in xs and y in ys:
				x_list.append(x); y_list.append(y)
	return x_list, y_list

def find_connected_neighbors(xc, yc, p_set):
	find_points = []
	for i in range(-1,2):
		for j in range(-1,2):
			if (xc+i, yc+j) in p_set and not (i==0 and j ==0):
				find_points.append((xc+i, yc+j))
	return find_points

map1 = map[:800,:800]
xs, ys = np.where(map1>0)
point_set = set()
for i in range(xs.shape[0]):
	point_set.add((xs[i],ys[i]))

# xlist, ylist = xs.tolist(), ys.tolist()
obj_list=[]; cur_obj=set(); obj_list.append(cur_obj)
# nb_pixels = 0
xc, yc = point_set.pop()
to_check_q = []; to_check_q.append((xc,yc))
while(len(point_set)>0 or len(to_check_q)>0):
# 	if nb_pixels == 0:
	while(len(to_check_q)>0):
		xc, yc = to_check_q.pop(0)
		cur_obj.add((xc,yc))
		neighbors = find_connected_neighbors(xc, yc, point_set)
		if len(neighbors)>0:
			to_check_q = to_check_q +neighbors
			point_set = point_set - set(neighbors)
	if len(point_set)>0:
		cur_obj=set(); obj_list.append(cur_obj); xc, yc = point_set.pop()
		to_check_q = []; to_check_q.append((xc,yc))

import scipy.ndimage as ndimage
mask_ext = np.zeros([map1.shape[0]+2, map1.shape[1]+2])
shp = map1.shape
mask_ext[1:-1,1:-1] = map1/2
masks = np.zeros(map1.shape+(9,))
for i in range(-1,2):
	for j in range(-1,2):
		masks[:,:,(i+1)*3+(j+1)]= mask_ext[i+1:i+shp[0]+1,j+1:j+shp[1]+1]
edge_mask = np.logical_and(np.sum(masks, axis = 2) < 9, np.sum(masks, axis = 2) >0)
ax = fig.add_subplot(2,2,1); ax.imshow(map1, cmap='Blues')
ax = fig.add_subplot(2,2,2); ax.imshow(edge_mask, cmap='Blues')
fill_mask = ndimage.morphology.binary_fill_holes(edge_mask)
ax = fig.add_subplot(2,2,3); ax.imshow(fill_mask, cmap='Blues')
ax = fig.add_subplot(2,2,4); ax.imshow(edge_mask*map1, cmap='Blues')


rows = 3; cols = 3
for i in range(rows):
	for j in range(cols):
		fig_indx = i*cols + j +1
		if fig_indx > len(obj_list)+1:
			break
		if fig_indx ==1:
			ax = fig.add_subplot(rows,cols,1); cax=ax.imshow(map1, cmap ='Blues'); fig.colorbar(cax); ax.set_title('Original map')
		else:
			obj_indices = list(obj_list[fig_indx-2])
			xlist =[]; ylist=[]
			for indx_tuple in obj_indices:
				xlist.append(indx_tuple[0]); ylist.append(indx_tuple[1])
			new_map = np.zeros(map1.shape)
			new_map[xlist,ylist] = 1
			ax = fig.add_subplot(rows,cols,fig_indx); cax=ax.imshow(new_map, cmap ='Blues'); fig.colorbar(cax)
plt.tight_layout(); plt.show()
			

def check_in_bag(xc,yc, obj_indices):
	is_inbag = False
	for i in range(-1,2):
		for j in range(-1,2):
			if (xc+i, yc+j) in obj_indices:
			 	is_inbag = True
			 	return is_inbag
	
	return is_inbag

# segment the mask into connected objects
fig = plt.figure(2);
xs, ys = np.where(map>0)
obj_list =[];
for i in range(len(xs)):
# 	if i ==0:
# 		cur_obj = Set(); obj_list.append(cur_obj)
	is_inbag = False
	for obj_indx in range(len(obj_list)):
		if check_in_bag(xs[i], ys[i], obj_list[obj_indx]):
			obj_list[obj_indx].add((xs[i],ys[i])); is_inbag = True
			break
	if not is_inbag:
		obj_list.append(set()); obj_list[-1].add((xs[i],ys[i]))

fig=plt.figure(3);plt.clf()
map_list = []
obj_indices = list(obj_list[0])
xlist =[]; ylist=[]
for indx_tuple in obj_indices:
	xlist.append(indx_tuple[0]); ylist.append(indx_tuple[1])
new_map = np.zeros(map.shape)
new_map[xlist,ylist] = 1
ax = fig.add_subplot(1,2,1); cax=ax.imshow(map, cmap ='Blues'); fig.colorbar(cax); ax.set_title('Original map')
ax = fig.add_subplot(1,2,2); cax=ax.imshow(new_map, cmap ='Blues'); fig.colorbar(cax); ax.set_title('Object mask')

rows = 4; cols = 6
for i in range(rows):
	for j in range(cols):
		fig_indx = i*cols + j +1
		if fig_indx > len(obj_list)+1:
			break
		if fig_indx ==1:
			ax = fig.add_subplot(rows,cols,1); cax=ax.imshow(map, cmap ='Blues'); fig.colorbar(cax); ax.set_title('Original map')
		else:
			obj_indices = list(obj_list[fig_indx-2])
			xlist =[]; ylist=[]
			for indx_tuple in obj_indices:
				xlist.append(indx_tuple[0]); ylist.append(indx_tuple[1])
			new_map = np.zeros(map.shape)
			new_map[xlist,ylist] = 1
			ax = fig.add_subplot(rows,cols,fig_indx); cax=ax.imshow(new_map, cmap ='Blues'); fig.colorbar(cax)
plt.tight_layout(); plt.show()