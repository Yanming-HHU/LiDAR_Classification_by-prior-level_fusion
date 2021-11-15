"""Raster feature from SegNet
Xiaoqiang Liu     2019/11/20
"""
import argparse
import numpy as np
import sys
import os
from tqdm import tqdm
from skimage import io
import pickle
import gc
import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from segnet import SegNet
from segnet_utils import count_sliding_window, sliding_window, grouper

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='/result_output/3_feature_level/data', help='path to the log file')
parser.add_argument('--model_path', default='./log/segnet/best_feature', help='path to segnet model file')
FLAGS = parser.parse_args()

LOG_DIR = FLAGS.log_dir
MODEL_PATH = FLAGS.model_path
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters
WINDOW_SIZE = (256, 256)   #patch size
IN_CHANNELS = 3
BATCH_SIZE = 16
N_FEATURES = 64
N_CLASSES = 6
WEIGHTS = torch.ones(N_CLASSES)   #This is weight for every class to handle imbalanced question

if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'feature_raster.log'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

def raster_feature(raster_file, split, model_path=None, batch_size=BATCH_SIZE, 
                   window_size=WINDOW_SIZE, stride=WINDOW_SIZE[0]):
    """Extraction raster feature using SegNet
    Args:
        raster_file: string, path to raster file
        type: string, training or testing
        model_path: string, path to model
        batch_size: int
        window_size: tuple
        stride: int
    Returns:
        None
    """
    img = np.asarray(io.imread(raster_file), dtype='float32')
    net = SegNet(in_channels=IN_CHANNELS, out_channels=N_CLASSES)
    if model_path is None:
        raise ValueError('No model')
    net.load_state_dict(torch.load(model_path))
    net.to(DEVICE)
    net.eval()
    
    raster_feature = np.zeros(img.shape[:2] + (N_FEATURES,))
    sliding_time = np.zeros(img.shape[:2])
    
    total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
    all_patch = sliding_window(img, step=stride, window_size=window_size)
    for i, coords in enumerate(tqdm(grouper(batch_size, all_patch), total=total, leave=False)):
        # Build the tensor
        image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
        image_patches = np.asarray(image_patches)
        image_patches = torch.from_numpy(image_patches).to(DEVICE)
        
        with torch.no_grad():
            _, features = net(image_patches)
        features = features.data.cpu().numpy()
        
        for feature, (x,y,w,h) in zip(features, coords):
            feature = feature.transpose((1,2,0))
            raster_feature[x:x+w, y:y+h] += feature
            sliding_time[x:x+w, y:y+h] += 1
           
    raster_feature = raster_feature.transpose(2,0,1) / sliding_time
    
    np.save(os.path.join(LOG_DIR, 'raster_feature_{}'.format(split)), raster_feature.transpose(1,2,0))

if __name__ == "__main__":

    os.system('cp '+__file__+' %s ' % (LOG_DIR))
    raster_files = ['../data/fill_train.tif', '../data/fill_test.tif']
    splits = ['train', 'test']
    for raster_file, split in zip(raster_files, splits):
        log_string('calculating the feature of {} data'.format(split))
        raster_feature(raster_file, split=split, stride=32, model_path=MODEL_PATH)
