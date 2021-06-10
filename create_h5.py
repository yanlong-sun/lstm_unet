import h5py
import os
import scipy.io as scio

action = 'test'      #  valid training test
slices_path = '../Dataset/' + action + '_data/' + action + '_data_mat/slices/'
masks_path = '../Dataset/' + action + '_data/' + action + '_data_mat/masks/'
if not os.path.exists('../Dataset/h5py/'):
    os.makedirs('../Dataset/h5py/')

slices_list = []
masks_list = []

slices_files = os.listdir(slices_path)
masks_files = os.listdir(masks_path)
slices_files.sort()
masks_files.sort()

for file in slices_files:
    if file[-4:] == '.mat':
        txt_file = scio.loadmat(slices_path + file)
        print(slices_path + file)
        slices_list.append(txt_file['images_to_save'])
for file in masks_files:
    if file[-4:] == '.mat':
        txt_file = scio.loadmat(masks_path + file)
        masks_list.append(txt_file['masks_to_save'])
print(len(slices_list))

h5py_file_name = '../Dataset/h5py/' + action + '_data.hdf5'
f = h5py.File(h5py_file_name, 'w')
f.create_dataset('X', data=slices_list)
f.create_dataset('Y', data=masks_list)
f.close()
