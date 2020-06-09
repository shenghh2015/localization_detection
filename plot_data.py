import numpy as np
import os
import glob
from skimage import io
import matplotlib.pyplot as plt

from helper_function import print_yellow, print_green, print_red

docker = True
if docker:
	dataset_dir = '../data/minidataset/'
else:
	dataset_dir = './data/Phase_fluo_Cells/0_FUCCI_Timelapse/'

phase_img_folder = dataset_dir+'f0_phase_cropped/'
fl1_img_folder = dataset_dir+'f0_fl1_cropped/'
fl2_img_folder = dataset_dir+'f0_fl2_cropped/'

phase_img_files = glob.glob(phase_img_folder+'*.tif')
fl1_img_files = glob.glob(fl1_img_folder+'*.tif')
fl2_img_files = glob.glob(fl2_img_folder+'*.tif')

nb_imgs = 6
for i in range(nb_imgs):
	print_yellow(os.path.basename(phase_img_files[i]))
	print_red(os.path.basename(fl1_img_files[i]))
	print_green(os.path.basename(fl2_img_files[i]))

# plt.ion(); 
fig = plt.figure(1)

index = 1
pha_file_name = os.path.basename(phase_img_files[index])
fl1_file_name = pha_file_name.replace('ch0', 'ch1').replace('mhilbert', 'mFL1')
fl2_file_name = pha_file_name.replace('ch0', 'ch2').replace('mhilbert', 'mFL2')
pha_img = io.imread(phase_img_folder+pha_file_name)
fl1_img = io.imread(fl1_img_folder+fl1_file_name)
fl2_img = io.imread(fl2_img_folder+fl2_file_name)
print(pha_file_name); print(fl1_file_name); print(fl2_file_name)
# fl1_img = io.imread(fl1_img_files[index])
# fl2_img = io.imread(fl2_img_files[index])

plt.clf()
x1, x2 = 500, 1500; y1, y2 = 1000, 1900
# pha_img_th = np.logical_and(pha_img>-0.1, pha_img<0.1)*pha_img
# fl1_img_th = np.logical_and(fl1_img>350, fl1_img < 550)*fl1_img
# fl2_img_th = np.logical_and(fl2_img>600, fl2_img < 1000)*fl2_img
ax=fig.add_subplot(3,3,1); cax=ax.imshow(pha_img, cmap='gray');fig.colorbar(cax);ax.set_title('Phase constrast');ax.set_ylabel('Image')
bx=fig.add_subplot(3,3,2); cbx=bx.imshow(fl1_img, cmap='gray');fig.colorbar(cbx);bx.set_title('Fluorscent 1')
cx=fig.add_subplot(3,3,3); ccx=cx.imshow(fl2_img, cmap='gray');fig.colorbar(ccx);cx.set_title('Fluorscent 2')
ax=fig.add_subplot(3,3,4); cax=ax.imshow(pha_img[y1:y2, x1:x2], cmap='gray');fig.colorbar(cax);ax.set_ylabel('Patch')
bx=fig.add_subplot(3,3,5); cbx=bx.imshow(fl1_img[y1:y2, x1:x2], cmap='gray');fig.colorbar(cbx);
cx=fig.add_subplot(3,3,6); ccx=cx.imshow(fl2_img[y1:y2, x1:x2], cmap='gray');fig.colorbar(ccx);
ax=fig.add_subplot(3,3,7); ax.hist(pha_img.flatten(), bins = 200);ax.set_title('Phase constrast')
ax=fig.add_subplot(3,3,8); ax.hist(fl1_img.flatten(), bins = 200);ax.set_title('Fluorscent 1')
ax=fig.add_subplot(3,3,9); ax.hist(fl2_img.flatten(), bins = 200);ax.set_title('Fluorscent 2')
plt.show()
plt.tight_layout()

## preprocessing
# bin_index = 25
# _, pha_bins = np.histogram(pha_img.flatten(), bins = 200); pha_lower, pha_upper = pha_bins[bin_index], pha_bins[-bin_index]
# _, fl1_bins = np.histogram(fl1_img.flatten(), bins = 200); fl1_lower, fl1_upper = fl1_bins[bin_index], fl1_bins[-bin_index]
# _, fl2_bins = np.histogram(fl2_img.flatten(), bins = 200); fl2_lower, fl2_upper = fl2_bins[bin_index], fl2_bins[-bin_index]


fig2 = plt.figure(2)
plt.clf()
pha_lower, pha_upper = pha_img.mean()-10*pha_img.std(), pha_img.mean()+10*pha_img.std()
fl1_lower, fl1_upper = fl1_img.mean()-10*fl1_img.std(), fl1_img.mean()+10*fl1_img.std()
# fl2_lower, fl2_upper = fl2_img.mean()-10*fl2_img.std(), fl2_img.mean()+10*fl2_img.std()
fl2_lower, fl2_upper = 850, 880
x1, x2 = 500, 1500; y1, y2 = 1000, 1900
pha_img_th = np.logical_and(pha_img>pha_lower, pha_img<pha_upper)*pha_img + (pha_img<=pha_lower)*(pha_lower) + (pha_img>=pha_upper)*pha_upper
fl1_img_th = np.logical_and(fl1_img>fl1_lower, fl1_img<fl1_upper)*fl1_img + (fl1_img<=fl1_lower)*(fl1_lower) + (fl1_img>=fl1_upper)*fl1_upper
fl2_img_th = np.logical_and(fl2_img>fl2_lower, fl2_img<fl2_upper)*fl2_img + (fl2_img<=fl2_lower)*(fl2_lower) + (fl2_img>=fl2_upper)*fl2_upper
ax=fig2.add_subplot(3,3,1); cax=ax.imshow(pha_img_th, cmap='gray');fig2.colorbar(cax);ax.set_title('Phase constrast');ax.set_ylabel('Image')
bx=fig2.add_subplot(3,3,2); cbx=bx.imshow(fl1_img_th, cmap='gray');fig2.colorbar(cbx);bx.set_title('Fluorscent 1')
cx=fig2.add_subplot(3,3,3); ccx=cx.imshow(fl2_img_th, cmap='gray');fig2.colorbar(ccx);cx.set_title('Fluorscent 2')
ax=fig2.add_subplot(3,3,4); cax=ax.imshow(pha_img_th[y1:y2, x1:x2], cmap='gray');fig2.colorbar(cax);ax.set_ylabel('Patch')
bx=fig2.add_subplot(3,3,5); cbx=bx.imshow(fl1_img_th[y1:y2, x1:x2], cmap='gray');fig2.colorbar(cbx);
cx=fig2.add_subplot(3,3,6); ccx=cx.imshow(fl2_img_th[y1:y2, x1:x2], cmap='gray');fig2.colorbar(ccx);
ax=fig2.add_subplot(3,3,7); ax.hist(pha_img_th.flatten(), bins = 200);ax.set_title('Phase constrast')
ax=fig2.add_subplot(3,3,8); ax.hist(fl1_img_th.flatten(), bins = 200);ax.set_title('Fluorscent 1')
ax=fig2.add_subplot(3,3,9); ax.hist(fl2_img_th.flatten(), bins = 200);ax.set_title('Fluorscent 2')
plt.show()
plt.tight_layout()

# def normal_0_1(img):
# 	return (img-img.min())/(img.max()-img.min())

# fig2 = plt.figure(2)
# plt.clf()
# x1, x2 = 500, 1500; y1, y2 = 1000, 1900
# ax=fig2.add_subplot(1,3,1); ax.hist(pha_img[y1:y2, x1:x2].flatten(), bins = 200);ax.set_title('Phase constrast')
# ax=fig2.add_subplot(1,3,2); ax.hist(fl1_img[y1:y2, x1:x2].flatten(), bins = 200);ax.set_title('Fluorscent 1')
# ax=fig2.add_subplot(1,3,3); ax.hist(fl2_img[y1:y2, x1:x2].flatten(), bins = 200);ax.set_title('Fluorscent 2')
# ax=fig2.add_subplot(1,3,1); ax.hist(pha_img.flatten(), bins = 200);ax.set_title('Phase constrast')
# ax=fig2.add_subplot(1,3,2); ax.hist(fl1_img.flatten(), bins = 200);ax.set_title('Fluorscent 1')
# ax=fig2.add_subplot(1,3,3); ax.hist(fl2_img.flatten(), bins = 200);ax.set_title('Fluorscent 2')
# plt.tight_layout()
