import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
import pydicom as dicom
import math
import re


class DCMDataLoader(object):
    def __init__(self, LDCT_image_path, NDCT_image_path, \
                 image_size=512, patch_size=64, depth=1,batch_size=32):
        # dicom file dir
        self.LDCT_image_path = LDCT_image_path
        self.NDCT_image_path = NDCT_image_path
        # image params
        self.image_size = image_size
        self.patch_size = patch_size
        self.depth = depth
        # training params
        self.batch_size = batch_size
        # CT slice name
        self.LDCT_image_name, self.NDCT_image_name = [], []
        # batch generator  prameters
        print('load case mean and std... ')
        self.case_mean_l = np.load("case_mean_l.npy")
        self.case_std_l = np.load("case_std_l.npy")
        self.patient_l = np.load("patient_l.npy")
        self.max_l = np.load("max_l.npy")
        self.min_l = np.load("min_l.npy")
        self.case_mean_h = np.load("case_mean_h.npy")
        self.case_std_h = np.load("case_std_h.npy")
        self.max_h = np.load("max_h.npy")
        self.min_h = np.load("min_h.npy")
        self.patient_h = np.load("patient_h.npy")

    def __call__(self, patent_no_list, LDCT_image_path, NDCT_image_path):
        p_LDCT = []
        p_NDCT = []
        # print('patent_no_list',patent_no_list)
        for patent_no in patent_no_list:
            print('patent_no', patent_no)
            P_LDCT_path = os.listdir(patent_no)
            print('P_LDCT_path', P_LDCT_path)
            for i in P_LDCT_path:
                if 'DIRFILE' in i:
                    P_LDCT_path.remove(i)
            patent_no_h = patent_no.replace(str(LDCT_image_path), str(NDCT_image_path))

            P_NDCT_path_r = os.listdir(patent_no_h)
            # if img not exit both in low_dose and high_dose path, the img will not be included in further analysis
            rem_p = []
            for j in range(len(P_LDCT_path)):
                if P_LDCT_path[j] not in P_NDCT_path_r:
                    rem_p.append(P_LDCT_path[j])
            for k in rem_p:
                P_LDCT_path.remove(k)

            P_NDCT_path = [i for i in range(len(P_LDCT_path))]
            for i in range(len(P_LDCT_path)):
                P_NDCT_path[i] = os.path.join(patent_no_h + '/' + P_LDCT_path[i])
                P_LDCT_path[i] = os.path.join(patent_no + '/' + P_LDCT_path[i])

            # load images
            # P_LDCT_path inculude all img path of one sample
            # CT slice name
            LDCT_slice_nm = self.get_slice_nm(P_LDCT_path, '{}_{}'.format(patent_no.split('\\')[-1], self.LDCT_image_path.split('\\')[-1]))
            NDCT_slice_nm = self.get_slice_nm(P_NDCT_path, '{}_{}'.format(patent_no_h.split('\\')[-1], self.NDCT_image_path.split('\\')[-1]))
            print('LDCT_slice_nm', LDCT_slice_nm)
            self.LDCT_image_name.extend(LDCT_slice_nm)
            self.NDCT_image_name.extend(NDCT_slice_nm)
            p_LDCT.append(P_LDCT_path)
            p_NDCT.append(P_NDCT_path)
            print('P_LDCT_path',len(P_LDCT_path))
            print('p_NDCT_path', len(P_NDCT_path))
            
        self.LDCT_images = np.concatenate(tuple(p_LDCT), axis=0)
        self.NDCT_images = np.concatenate(tuple(p_NDCT), axis=0)
        # image index
        self.LDCT_index, self.NDCT_index = list(range(len(self.LDCT_images))), list(range(len(self.NDCT_images)))
        np.random.seed(10)
        np.random.shuffle(self.LDCT_images)
        np.random.seed(10)
        np.random.shuffle(self.NDCT_images)

    def normalize(self, img, max_=3072, min_=-1024):
        img = img.astype(np.float32)
        img[img > max_] = max_
        img[img < min_] = min_
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        return img

    def normalization(self, path, case_mean_all, case_std_all, max_n, min_n, patient_list):
        dicom_array = dicom.read_file(path)
        dicom_image = self.get_pixels_hu(dicom_array)
        dicom_image[dicom_image > max_n] = max_n
        dicom_image[dicom_image < min_n] = min_n
        patient = re.split(r'[\\/]', path)[-2]
        patient_list = patient_list.tolist()
        index_p = patient_list.index(patient)
        mean_p = case_mean_all[index_p]
        std_p = case_std_all[index_p]
        dicom_image = (dicom_image - min_n)/(max_n-min_n)
        # dicom_image = self.normalize(dicom_image)
        # plt.imshow(dicom_image)
        # plt.show()
        return dicom_image

    def get_slice_nm(self, P_LDCT_path, patent_no):
        digit = 4
        slice_nm = []
        for slice_number in range(len(P_LDCT_path)):
            name = P_LDCT_path[slice_number].split('\\')[-1]
            # sorted(idx), sorted(d_idx)  -> [1, 10, 2], [ 0001, 0002, 0010]
            s_idx = str(slice_number)
            d_idx = '0' * (digit - len(s_idx)) + s_idx
            slice_nm.append(patent_no + '_' + name + '_' + d_idx)
        return slice_nm


    def get_pixels_hu(self, slice, pre_fix_nm=''):
        image = slice.pixel_array
        image = image.astype(np.int16)
        image[image == -2000] = 0
        intercept = slice.RescaleIntercept
        slope = slice.RescaleSlope
        if slope != 1:
            image = slope * image.astype(np.float32)
            image = image.astype(np.int16)
        image += np.int16(intercept)
        return np.array(image, dtype=np.int16)

    def test_img(self, LDCT_slice, NDCT_slice):
        LDCT_image = self.normalization(LDCT_slice, self.case_mean_l, self.case_std_l, self.max_l, self.min_l, self.patient_l)
        NDCT_image = self.normalization(NDCT_slice, self.case_mean_h, self.case_std_h, self.max_h, self.min_h, self.patient_h)
        return LDCT_image, NDCT_image

    # WGAN_VGG, RED_CNN
    def get_randam_patches(self, LDCT_slice, NDCT_slice, patch_size, whole_size=512):
        # truncate & normalization
        LDCT_image = self.normalization(LDCT_slice, self.case_mean_l, self.case_std_l, self.max_l, self.min_l, self.patient_l)
        NDCT_image = self.normalization(NDCT_slice, self.case_mean_h, self.case_std_h, self.max_h, self.min_h, self.patient_h)
        whole_h = whole_w = whole_size
        h = w = patch_size
        # patch image range
        hd, hu = h // 2, int(whole_h - np.round(h / 2))
        wd, wu = w // 2, int(whole_w - np.round(w / 2))
        h_pc = np.random.choice(range(hd, hu + 1))
        w_pc = np.random.choice(range(wd, wu + 1))
        if len(LDCT_image.shape) == 3:  # 3d patch
            LDCT_patch = LDCT_image[:, h_pc - hd: int(h_pc + np.round(h / 2)), w_pc - wd: int(w_pc + np.round(h / 2))]
            NDCT_patch = NDCT_image[:, h_pc - hd: int(h_pc + np.round(h / 2)), w_pc - wd: int(w_pc + np.round(h / 2))]
        else:  # 2d patch
            LDCT_patch = LDCT_image[h_pc - hd: int(h_pc + np.round(h / 2)), w_pc - wd: int(w_pc + np.round(h / 2))]
            NDCT_patch = NDCT_image[h_pc - hd: int(h_pc + np.round(h / 2)), w_pc - wd: int(w_pc + np.round(h / 2))]
        return LDCT_patch, NDCT_patch

    def preproc_input(self, args):
        n_batches = int(len(self.LDCT_index)/(self.batch_size))
        print('n_batches', n_batches)
        for i in range(n_batches - 1):
            batch_LDCT = self.LDCT_images[i * self.batch_size:(i + 1) * self.batch_size]
            batch_NDCT = self.NDCT_images[i * self.batch_size:(i + 1) * self.batch_size]
            LDCT_imgs, NDCT_imgs = [], []
            for img_index in range(len(batch_LDCT)):
                mean_n = -1
                if self.patch_size != self.image_size:
                    while mean_n < -0.4:
                        LDCT_image, NDCT_image = self.get_randam_patches(batch_LDCT[img_index],batch_NDCT[img_index], args.patch_size)
                        mean_n = np.mean(LDCT_image)
                else:
                    LDCT_image = self.normalization(batch_LDCT[img_index], self.case_mean_l, self.case_std_l,
                                                self.patient_l)
                    NDCT_image = self.normalization(batch_NDCT[img_index], self.case_mean_h, self.case_std_h,
                                                self.patient_h)
                # print('mean_low', np.mean(LDCT_image))
                # plt.imshow(LDCT_image, cmap='gray')
                # plt.show()
                # plt.imshow(NDCT_image, cmap='gray')
                # plt.show()

                LDCT_imgs.append(np.expand_dims(LDCT_image, axis=-1))
                NDCT_imgs.append(np.expand_dims(NDCT_image, axis=-1))
            yield np.array(LDCT_imgs), np.array(NDCT_imgs)

# psnr
def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def tf_psnr(img1, img2, PIXEL_MAX=255.0):
    mse = tf.reduce_mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * log10(PIXEL_MAX / tf.sqrt(mse))

def psnr(img1, img2, PIXEL_MAX=255.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# save mk img
def save_image(LDCT, NDCT, output_, save_dir='.', max_=1, min_=0):
    f, axes = plt.subplots(2, 3, figsize=(30, 20))

    axes[0, 0].imshow(LDCT, cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[0, 1].imshow(NDCT, cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[0, 2].imshow(output_, cmap=plt.cm.gray, vmax=max_, vmin=min_)

    axes[1, 0].imshow(NDCT.astype(np.float32) - LDCT.astype(np.float32), cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[1, 1].imshow(NDCT - output_, cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[1, 2].imshow(output_ - LDCT, cmap=plt.cm.gray, vmax=max_, vmin=min_)

    axes[0, 0].title.set_text('LDCT image')
    axes[0, 1].title.set_text('NDCT image')
    axes[0, 2].title.set_text('output image')

    axes[1, 0].title.set_text('NDCT - LDCT  image')
    axes[1, 1].title.set_text('NDCT - outupt image')
    axes[1, 2].title.set_text('output - LDCT  image')
    if save_dir != '.':
        f.savefig(save_dir)
        plt.close()

    # ---------------------------------------------------

# argparser string -> boolean type
def ParseBoolean(b):
    b = b.lower()
    if b == 'true':
        return True
    elif b == 'false':
        return False
    else:
        raise ValueError('Cannot parse string into boolean.')

# argparser string -> boolean type
def ParseList(l):
    return l.split(',')
