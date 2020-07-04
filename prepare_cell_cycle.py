import numpy as np
import os
import glob
from skimage import io
import matplotlib.pyplot as plt
from helper_function import print_yellow, print_green, print_red, generate_folder
import scipy.ndimage as ndimage

docker = False
dataset_dir='../data/minidataset/' if docker else './data/Phase_fluo_Cells/0_FUCCI_Timelapse/'

output_dir = '/home/sh38/Mask_RCNN/datasets/cell_cycle/'

phase_img_folder = dataset_dir+'f0_phase_cropped/'
fl1_img_folder = dataset_dir+'f0_fl1_cropped/'
fl2_img_folder = dataset_dir+'f0_fl2_cropped/'
combined_masks = dataset_dir+'f0_combined_masks/'

phase_img_files = glob.glob(phase_img_folder+'*.tif')
fl1_img_files = glob.glob(fl1_img_folder+'*.tif')
fl2_img_files = glob.glob(fl2_img_folder+'*.tif')
mask_files = glob.glob(combined_masks+'*.tif')

image_ids = []
for i in range(1000):
    phase_img_file = phase_img_files[i]
    file_name = os.path.basename(phase_img_file)
    digit_str = file_name.split('_')[1][1:]
    if int(digit_str)<=25:
        image_ids.append(i)
print('The number of good examples: {}'.format(len(image_ids)))

print(len(phase_img_files)); print(len(fl1_img_files)); 
print(len(fl2_img_files));print(len(mask_files));

def find_connected_neighbors(xc, yc, p_set):
	find_points = []
	for i in range(-1,2):
		for j in range(-1,2):
			if (xc+i, yc+j) in p_set and not (i==0 and j ==0):
				find_points.append((xc+i, yc+j))
	return find_points

for image_indx in range(len(image_ids)):
	# Load the map
	image_id = image_ids[image_indx]
	pha_file_name = os.path.basename(phase_img_files[image_id])
	mask_file_name = 'm_'+pha_file_name.replace('ch0', 'ch4').replace('mhilbert', 'mFL4')
	pha_img = io.imread(phase_img_folder+pha_file_name)
	map = io.imread(combined_masks+mask_file_name)
	if image_indx%10==0:
		print_yellow('The {}-th image'.format(image_indx))
		print_yellow(mask_files[image_id])
		print_green(combined_masks+mask_file_name)
	mask_list = []

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
			if np.sum(mask)>500:
				mask_list.append(mask) ## add to the mask list

	# Check the mask
	bin_map = map > 0; 
	recon_mask =np.zeros(bin_map.shape)
	for i in range(len(mask_list)):
		if np.sum(mask_list[i])>500:
			recon_mask = recon_mask + mask_list[i]*map
	print_green(np.sum(map)); print_red(np.sum(np.uint8(recon_mask)))
	print_green(np.unique(map)); print_red(np.unique(np.uint8(recon_mask)))

	# Save images and masks
	rgb_pha_img = np.uint8(255*(pha_img-np.min(pha_img))/(np.max(pha_img)-np.min(pha_img)))
	rgb_pha_img = np.concatenate([rgb_pha_img.reshape(rgb_pha_img.shape+(1,)), rgb_pha_img.reshape(rgb_pha_img.shape+(1,)), rgb_pha_img.reshape(rgb_pha_img.shape+(1,))], axis =2)
	data_folder = os.path.join(output_dir, '{:04d}'.format(image_indx)); new_image_id = '{:04d}'.format(image_indx)
	generate_folder(data_folder+'/images'); generate_folder(data_folder+'/masks'); generate_folder(data_folder+'/GT')
	generate_folder(data_folder+'/original_GT')
	io.imsave(data_folder+'/images/{}.png'.format(new_image_id),rgb_pha_img)
	io.imsave(data_folder+'/GT/{}.png'.format(new_image_id),recon_mask)
	io.imsave(data_folder+'/original_GT/{}.png'.format(new_image_id),map)
	for i in range(len(mask_list)):
		io.imsave(data_folder+'/masks/mask_{:04d}.png'.format(i),mask_list[i]*map)