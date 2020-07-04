import os
import numpy as np
import glob
from skimage import io

dataset = 'val'
dataset_folder = './data/Phase_fluo_Cells/phase_cells/{}'.format(dataset)
img_folders = glob.glob(dataset_folder+'/*')

recon_maps = []
gt_maps = []
dim = 832
for img_folder in img_folders:
	img_id = os.path.basename(img_folder)
	img = io.imread(img_folder+'/images/{}.png'.format(img_id))
	masks = glob.glob(img_folder+'/masks/*.png')
	recon_map = np.zeros(img.shape[:2])
	for mask in masks:
		recon_map = recon_map + io.imread(mask)
	gt_map = io.imread(img_folder+'/GT/{}.png'.format(img_id))
	recon_maps.append(recon_map); gt_maps.append(gt_map) 
recon_map_arr = np.concatenate(recon_maps)
gt_map_arr = np.concatenate(gt_maps)

print('Err:{}'.format(np.mean(np.abs(recon_map_arr -gt_map_arr))))