"""Utils for segnet_train.py
modified based on https://github.com/nshaud/DeepNetsForEO/blob/master/SegNet_PyTorch_v2.ipynb

Xiaoqiang Liu     2019/11/20
"""

import numpy as np
from skimage import io
import random
import itertools
import os
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn.functional as F

LABELS = ['imp sur','buildings', 'low veg', 'trees', 'others']
N_CLASSES = len(LABELS)
WEIGHTS = torch.ones(N_CLASSES)
CACHE = False

MAIN_FOLDER = '../data/VaihingenRaster/'
DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top/top_mosaic_09cm_area{}_noBoundary.tif'

# Dataset class
class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_file=DATA_FOLDER, label_files=LABEL_FOLDER, cache=False, augmentation=True):
        super(ISPRS_dataset, self).__init__()
        
        self.augmentation = augmentation
        self.cache = cache
        
        #List of files
        self.data_files = [DATA_FOLDER.format(id) for id in ids]
        self.label_files = [LABEL_FOLDER.format(id) for id in ids]
        
        # Sanity check: raise an error if some files do not exist
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file!'.format(f))
        
        # initialize cache dicts
        self.data_cache_ = {}
        self.label_cache_ = {}
    
    def __len__(self):
        # Default epoch size is 784 samples
        # because all image can be split by window(256, 256) into 1046
        # the train use 12/16 samples
        return 784 
    
    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
        
        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:,::-1]
                else:
                    array = array[:,:,::-1]
            results.append(np.copy(array))
        return tuple(results)
    
    def get_random_pos(self, img, window_shape=(256, 256)):
        """Extract of 2D random pathc of shape window_shape in the image""" 
        w, h = window_shape
        W, H = img.shape[-2:]
        x1 = random.randint(0, W-w-1)
        x2 = x1+w
        y1 = random.randint(0, H-h-1)
        y2 = y1+h
        return x1, x2, y1, y2
    
    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files)-1)
        
        # if the tile hsan't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalize in [0, 1]
            data = np.asarray(io.imread(self.data_files[random_idx]).transpose((2,0,1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data
        
        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label
        
        # Get a random patch
        x1, x2, y1, y2 = self.get_random_pos(data)
        data_p = data[:, x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]
        
        #Data augmentation
        data_p, label_p = self.data_augmentation(data_p, label_p)
        
        #Return the torch Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p))

# define the standard ISPRS color palette
# palette = {0 : (255, 255, 255), # Impervious surfaces (white)
#            1 : (0, 0, 255),     # Buildings (blue)
#            2 : (0, 255, 255),   # Low vegetation (cyan)
#            3 : (0, 255, 0),     # Trees (green)
#            4 : (255, 255, 0),   # Cars (yellow)
#            5 : (255, 0, 0),     # Clutter (red)
#            6 : (0, 0, 0)}       # Undefined (black)

#Color and label convert using dictionary
palette = {0 : (169, 170, 126), # Impervious surfaces (white)
           1 : (255, 171, 127),     # Buildings (blue)
           2 : (170, 255, 126),   # Low vegetation (cyan)
           3 : (0, 170, 0),     # Trees (green)
           4 : (255, 255, 127)}       # others include Cars, Clutter and Undefined

invert_palette = {(255, 255, 255): 0,  # Impervious surfaces (white)
                  (0, 0, 255): 1,      # Buildings (blue)
                  (0, 255, 255): 2,    # Low vegetation (cyan)
                  (0, 255, 0): 3,      # Trees (green)
                  (255, 255, 0): 4,    # Cars (yellow) to others
                  (255, 0, 0): 4,      # Clutter (red) to others
                  (0, 0, 0): 4}        # Undefined (black) to others

def convert_to_color(arr_2d, palette=palette):
    """Numeric labels to RGB-color encoding"""
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1],3), dtype=np.uint8)
    
    for c, i in palette.items():
        m = arr_2d==c
        arr_3d[m] = i
        
    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """RGB-color encoding to grayscale labels"""
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    
    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1,1,3), axis=2)
        arr_2d[m] = i
    tmp = arr_2d == 5
    arr_2d[tmp] = 4
    tmp = arr_2d == 6
    arr_2d[tmp] = 4
    return arr_2d

# Utils
def cal_metric(gts, preds, log_string=None):
    """
    Args:
        gts: List<int>, prediction of data
        preds: List<int>, ground truth
        log_string, fuction for Log record
    Returns:
        accuracy: float, overall accuracy
    """
    tmp = precision_recall_fscore_support(gts, preds)
    if log_string is not None:
        log_string(str(tmp))
    return precision_recall_fscore_support(gts, preds, average='weighted')[1]

def sliding_window(top, step=256, window_size=(256,256)):
    """Slide a window_shape window across the image with a stride of step"""
    for x in range(0, top.shape[0], step):
        if x+window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=256, window_size=(256,256)):
    """Count the number of windows in an image"""
    tmp = np.ceil(float(top.shape[0])/step) * np.ceil(float(top.shape[1])/step)
    return int(tmp)
    
def grouper(n, iterable):
    """Browse an iterator by chunk of n elements"""
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk