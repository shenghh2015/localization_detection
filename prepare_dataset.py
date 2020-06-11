import numpy as np
import os
import glob
from skimage import io
import matplotlib.pyplot as plt
import csv
import scipy.ndimage as ndimage

from helper_function import print_yellow, print_green, print_red, generate_folder

docker = True
if docker:
	dataset_dir = '../data/Data_registered/'
	output_dir = '../data/Processed/'
else:
	dataset_dir = './data/Phase_fluo_Cells/Data_registered/'
	output_dir = '../data/Processed/'

dataset = 'train';  output_dir=output_dir+dataset+'/'
csv_file = dataset_dir+'{}.csv'.format(dataset)
hrSLM_file = dataset_dir+'{}_hrSLIM.csv'.format(dataset)

## obtain the files
img_fnames =[]; map_fnames = []
with open(csv_file, newline='') as f:
	csv_reader = csv.reader(f)
	for row in csv_reader:
		img_fnames.append(row[0]); map_fnames.append(row[1])

img_fnames1= img_fnames; map_fnames1 = map_fnames

img_fnames =[]; map_fnames = []
with open(hrSLM_file, newline='') as f:
	csv_reader = csv.reader(f)
	for row in csv_reader:
		img_fnames.append(row[0]); map_fnames.append(row[1])

def find_connected_neighbors(xc, yc, p_set):
	find_points = []
	for i in range(-1,2):
		for j in range(-1,2):
			if (xc+i, yc+j) in p_set and not (i==0 and j ==0):
				find_points.append((xc+i, yc+j))
	return find_points

# nb_images = len(img_fnames)
nb_images = 10
for image_indx in range(nb_images):
	# Load the map
# 	image_indx=0; 
	map = io.imread(dataset_dir+map_fnames[image_indx]); mask_list = []

	# Generate the edge masks for all objects
	uni_values = np.unique(map)
	for cls_indx in range(1,len(uni_values)):
		map_layer = map == uni_values[cls_indx]
		mask_ext = np.zeros([map_layer.shape[0]+2, map_layer.shape[1]+2])
		shp = map_layer.shape
		mask_ext[1:-1,1:-1] = map_layer
		masks = np.zeros(map_layer.shape+(9,))
		for i in range(-1,2):
			for j in range(-1,2):
				masks[:,:,(i+1)*3+(j+1)]= mask_ext[i+1:i+shp[0]+1,j+1:j+shp[1]+1]
		edge_mask = np.logical_and(np.sum(masks, axis = 2) < 9, np.sum(masks, axis = 2) >0)*map_layer
		fill_mask = ndimage.morphology.binary_fill_holes(edge_mask)

		# Generate the edge mask for each object individually
		xs, ys = np.where(edge_mask>0)
		point_set = set()
		for i in range(xs.shape[0]):
			point_set.add((xs[i],ys[i]))

		obj_list=[]; cur_obj=set(); obj_list.append(cur_obj)
		xc, yc = point_set.pop()
		to_check_q = []; to_check_q.append((xc,yc))
		while(len(point_set)>0 or len(to_check_q)>0):
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

		# Fill the hole to get the mask for each object
		for i in range(len(obj_list)):
			mask = np.zeros(map.shape)
			for xy_tuple in list(obj_list[i]):
				mask[xy_tuple]=1
			mask = ndimage.morphology.binary_fill_holes(mask)
			mask_list.append(mask) ## add to the mask list

	# Plot the mask generation results
	# plt.ion();fig = plt.figure(1)
	# rows = 4; cols = 5
	# for i in range(rows):
	# 	for j in range(cols):
	# 		fig_indx = i*cols + j +1
	# 		if fig_indx > len(mask_list)+1:
	# 			break
	# 		if fig_indx ==1:
	# 			ax = fig.add_subplot(rows,cols,1); cax=ax.imshow(map, cmap ='Blues'); #fig.colorbar(cax); #ax.set_title('Original map')
	# 		else:
	# 			ax = fig.add_subplot(rows,cols,fig_indx); cax=ax.imshow(mask_list[fig_indx-2], cmap ='Blues'); #fig.colorbar(cax)
	# #plt.tight_layout(); 
	# plt.show()

	# Check the mask
	bin_map = map > 0; 
	recon_mask =np.zeros(bin_map.shape)
	for i in range(len(mask_list)):
		recon_mask = recon_mask + mask_list[i]
	print_green(np.sum(bin_map), np.sum(recon_mask))

	# Save images and masks
	hrSLM_img = io.imread(dataset_dir+img_fnames[image_indx]); mSLM_img = io.imread(dataset_dir+img_fnames1[image_indx])
	hrSLM_img_n = np.uint8(255*(hrSLM_img-np.min(hrSLM_img))/(np.max(hrSLM_img)-np.min(hrSLM_img)))
	mSLM_img_n = np.uint8(255*(mSLM_img-np.min(mSLM_img))/(np.max(mSLM_img)-np.min(mSLM_img)))
	shp = hrSLM_img.shape
	SLM_img = np.concatenate([hrSLM_img_n.reshape(shp+(1,)), mSLM_img_n.reshape(shp+(1,)), mSLM_img_n.reshape(shp+(1,))], axis =2)
	# ax = fig.add_subplot(2,2,1); cax=ax.imshow(hrSLM_img_n); 
	# ax = fig.add_subplot(2,2,2); cax=ax.imshow(mSLM_img_n);
	# ax = fig.add_subplot(2,2,3); cax=ax.imshow(SLM_img); 
	# ax = fig.add_subplot(2,2,4); cax=ax.imshow(recon_mask);
	data_folder = os.path.join(output_dir, '{:04d}'.format(image_indx)); image_id = '{:04d}'.format(image_indx)
	generate_folder(data_folder+'/images'); generate_folder(data_folder+'/masks'); generate_folder(data_folder+'/GT')
	io.imsave(data_folder+'/images/{}.png'.format(image_id),SLM_img)
	io.imsave(data_folder+'/GT/{}.png'.format(image_id),map)
	for i in range(len(mask_list)):
		io.imsave(data_folder+'/masks/mask_{:04d}.png'.format(i),mask_list[i]*map)
	# 	print(np.unique(mask_list[i]*map))