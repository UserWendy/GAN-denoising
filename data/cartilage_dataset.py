import os.path
import random
import torchvision.transforms as transforms
import torch
import scipy.io as io
from dataloader.base_dataset import BaseDataset
import numpy as np
import nibabel as nib
from util.image_augmentation import image_augmentation, image_augmentation_ae
from scipy import ndimage


class CartilageDataset(BaseDataset):

    def __init__(self, opt, training_id):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.training_id = training_id
        self.opt = opt
        self.root = opt.dataroot
        self.predict_root = opt.predict_root

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass

    def load_data(self, n_volume_num):
        # the load function from nib load the dimension z first
        n_step = self.opt.dataset_step

        str_file_name = self.root + '/%d/Mask.nii.gz' % (n_volume_num + 1)
        img = nib.load(str_file_name)
        self.mask = img.get_fdata().astype(np.float32)
        self.mask = self.mask[0::n_step, 0::n_step, 0::n_step]
        if self.opt.model == 'auto_encoder':
            self.volume = self.mask.copy()
        else:
            str_file_name = self.root + '/%d/Img.nii.gz' % (n_volume_num + 1)
            img = nib.load(str_file_name)
            volume = img.get_fdata()
            volume = volume.astype(np.float32)
            if self.opt.norm_mode == 'mean_std':
                self.volume = (volume - volume.mean()) / volume.std()
            else:
                self.volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
            self.volume = self.volume[0::n_step, 0::n_step, 0::n_step]
        # # test
        # filename = 'C:/data/test/img.raw'
        # self.volume.squeeze().tofile(filename)
        # filename = 'C:/data/test/mask.raw'
        # self.mask.squeeze().tofile(filename)

        print('Volume Reading....')
        # extract the index of the roi
        self.convert_batch(n_volume_num)
        # pad the volume and mask
        # n_padding_value = int(self.opt.patch_size / 2) + 16
        npad = ((self.opt.patch_size_w // 2, self.opt.patch_size_w // 2),
                (self.opt.patch_size_h // 2, self.opt.patch_size_h // 2),
                (self.opt.patch_size_d // 2, self.opt.patch_size_d // 2))
        volume_padded = np.pad(self.volume, pad_width=npad, mode='edge')
        mask_padded = np.pad(self.mask, pad_width=npad, mode='edge')
        self.volumes.append(volume_padded)
        self.masks.append(mask_padded)

    @staticmethod
    def extract_index(ind_pos, num_seed, n_volume_num):
        num_ind = len(ind_pos[0])
        z = ind_pos[0]
        y = ind_pos[1]
        x = ind_pos[2]
        sample_id = random.sample(range(num_ind), num_seed)
        index_list = []
        for i in range(num_seed):
            ids = sample_id[i]
            index = [n_volume_num, z[ids], y[ids], x[ids]]
            index_list.append(index)
        return index_list

    def convert_batch(self, n_volume_num):
        '''this function is aimed to sample the training patch'''
        mask_index = self.mask.copy()
        # create the boundary band labeled 5
        roi_ind_fem_cart = np.where(mask_index == 2)
        roi_ind_tib_cart = np.where(mask_index == 4)
        ind_foreground = np.where(mask_index == 1)
        mask_band = np.zeros_like(mask_index, dtype=np.uint8)
        mask_band[ind_foreground] = 1
        struct = ndimage.generate_binary_structure(3, 3)
        mask_dilated = ndimage.binary_dilation(mask_band, structure=struct, iterations=5).astype(mask_band.dtype)
        mask_eroded = ndimage.binary_erosion(mask_band, structure=struct, iterations=5).astype(mask_band.dtype)
        mask_band = mask_dilated - mask_eroded
        ind_band_fem = np.where(mask_band == 1)
        mask_index[ind_band_fem] = 5
        # create the boundary band labeled 6
        ind_foreground = np.where(mask_index == 3)
        mask_band = np.zeros_like(mask_index, dtype=np.uint8)
        mask_band[ind_foreground] = 1
        struct = ndimage.generate_binary_structure(3, 3)
        mask_dilated = ndimage.binary_dilation(mask_band, structure=struct, iterations=5).astype(mask_band.dtype)
        mask_eroded = ndimage.binary_erosion(mask_band, structure=struct, iterations=5).astype(mask_band.dtype)
        mask_band = mask_dilated - mask_eroded
        ind_band_tib = np.where(mask_band == 1)
        mask_index[ind_band_tib] = 6

        roi_ind_back = np.where(mask_index == 0)
        roi_ind_fem = np.where(mask_index == 1)
        roi_ind_tib = np.where(mask_index == 3)

        random.seed(1) # guarantee the repeat result
        # boundary femur band
        index = self.extract_index(ind_band_fem, self.opt.seed_size-self.opt.seed_change, n_volume_num)
        self.index_list.extend(index)

        # boundary tibia band
        index = self.extract_index(ind_band_tib, self.opt.seed_size-self.opt.seed_change, n_volume_num)
        self.index_list.extend(index)

        # background
        index = self.extract_index(roi_ind_back, self.opt.seed_size, n_volume_num)
        self.index_list.extend(index)

        # femur
        index = self.extract_index(roi_ind_fem, self.opt.seed_change, n_volume_num)
        self.index_list.extend(index)

        # tibia
        index = self.extract_index(roi_ind_tib, self.opt.seed_change, n_volume_num)
        self.index_list.extend(index)

        # femur cartilage
        index = self.extract_index(roi_ind_fem_cart, self.opt.seed_size, n_volume_num)
        self.index_list.extend(index)

        # tibia cartilage
        index = self.extract_index(roi_ind_tib_cart, self.opt.seed_size, n_volume_num)
        self.index_list.extend(index)

    def __getitem__(self, index):
        # remember z means the third dim in array, but width in image
        ind = self.index_list[index]
        n_volume_num = ind[0]
        z = ind[1]
        y = ind[2]
        x = ind[3]

        volume = self.volumes[n_volume_num]
        mask = self.masks[n_volume_num]

        n_pad_val_w = self.opt.patch_size_w // 2
        n_pad_val_h = self.opt.patch_size_h // 2
        n_pad_val_d = self.opt.patch_size_d // 2

        n_patch_w = self.opt.patch_size_w // 2
        n_patch_h = self.opt.patch_size_h // 2
        n_patch_d = self.opt.patch_size_d // 2

        position = [z + n_pad_val_w, y + n_pad_val_h, x + n_pad_val_d]
        volume_patch = volume[position[0] - n_patch_w:position[0] + n_patch_w,
                       position[1] - n_patch_h:position[1] + n_patch_h,
                       position[2] - n_patch_d:position[2] + n_patch_d]
        mask_patch = mask[position[0] - n_patch_w:position[0] + n_patch_w,
                    position[1] - n_patch_h:position[1] + n_patch_h,
                    position[2] - n_patch_d:position[2] + n_patch_d]

        # volume_patch = transforms.ToTensor()(volume_patch)
        # mask_patch = transforms.ToTensor()(mask_patch)
        # volume_patch = volume_patch.unsqueeze(0)
        # mask_patch = mask_patch.unsqueeze(0)
        if self.opt.model == 'unet3d' and self.opt.model_unet != 'full':
            volume_patch, mask_patch = image_augmentation(volume_patch, mask_patch, self.opt)
            return {'volume': volume_patch, 'mask': mask_patch, 'mode': 'train'}
        else:
            volume_patch, mask_patch = image_augmentation_ae(volume_patch, mask_patch, self.opt)
            return {'volume': volume_patch, 'mask': mask_patch, 'mode': 'train'}



    def __len__(self):
        return max(len(self.training_id), len(self.index_list))

    def name(self):
        return 'BlackBloodDataset'
