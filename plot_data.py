import numpy as np
import os
import glob
from skimage import io

from helper_function import print_yellow, print_green, print_red

docker = False
if docker:
	dataset_dir = ''
else:
	dataset_dir = './data/Phase_fluo_Cells/0_FUCCI_Timelapse/'

phase_img_folder = dataset_dir+'f0_phase_cropped/'
fl1_img_folder = dataset_dir+'f0_fl1_cropped/'
fl2_img_folder = dataset_dir+'f0_fl2_cropped/'

phase_img_files = glob.glob(phase_img_folder+'*.tif')
fl1_img_files = glob.glob(fl1_img_folder+'*.tif')
fl2_img_files = glob.glob(fl2_img_folder+'*.tif')

nb_imgs = 10
for i in range(nb_imgs):
	print_yellow(os.path.basename(phase_img_files[i]))
	print_red(os.path.basename(fl1_img_files[i]))
	print_green(os.path.basename(fl2_img_files[i]))

index = 0
pha_img = io.imread(phase_img_files[0])
fl1_img = io.imread(fl1_img_files[0])
fl2_img = io.imread(fl2_img_files[0])