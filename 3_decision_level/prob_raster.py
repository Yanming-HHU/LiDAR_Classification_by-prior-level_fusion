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
parser.add_argument('--output_dir', default='/result_output/4_decision_level/data', help='path to the output')
parser.add_argument('--model_path', default='./log/segnet/best_feature', help='path to segnet model file')
parser.add_argument('--input_path', default='../data/Vaihingen3D_testing.tif')
FLAGS = parser.parse_args()

OUTPUT_DIR = FLAGS.output_dir
MODEL_PATH = FLAGS.model_path
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Parameters
WINDOW_SIZE = (256, 256)   #patch size
IN_CHANNELS = 3
BATCH_SIZE = 16
N_FEATURES = 64
LABELS = ['low veg','tree', 'imp sur', 'car', 'building', 'background']
N_CLASSES = len(LABELS)

if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
LOG_FOUT = open(os.path.join(OUTPUT_DIR, 'prob_raster.log'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

def raster_prob(raster_file, model_path=None, batch_size=BATCH_SIZE, 
                   window_size=WINDOW_SIZE, stride=WINDOW_SIZE[0]):
    """inference raster probabity using SegNet
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
    img = 1.0/255 * np.asarray(io.imread(raster_file), dtype='float32')
    net = SegNet(in_channels=IN_CHANNELS, out_channels=N_CLASSES)
    if model_path is None:
        raise ValueError('No model')
    net.load_state_dict(torch.load(model_path))
    net.to(DEVICE)
    net.eval()
    
    raster_prob = np.zeros(img.shape[:2] + (N_CLASSES,))
       
    total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
    all_patch = sliding_window(img, step=stride, window_size=window_size)
    for i, coords in enumerate(tqdm(grouper(batch_size, all_patch), total=total, leave=False)):
        # Build the tensor
        image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
        image_patches = np.asarray(image_patches)
        image_patches = torch.from_numpy(image_patches).to(DEVICE)
        
        with torch.no_grad():
            prob_patches, _ = net(image_patches)
        prob_patches = torch.exp(prob_patches)
        prob_patches = prob_patches.data.cpu().numpy()
        
        for prob, (x,y,w,h) in zip(prob_patches, coords):
            prob = prob.transpose((1,2,0))
            raster_prob[x:x+w, y:y+h] += prob    
    
    return raster_prob.transpose(2,0,1) / np.sum(raster_prob, axis=-1)

if __name__ == "__main__":

    os.system('cp '+__file__+' %s ' % (OUTPUT_DIR))
    raster_file = '../data/fill_{}.tif' 
    log_string('calculating the probility:')
    
    for split in ['train', 'test']:
        print('this is for {}'.format(split))
        prob = raster_prob(raster_file.format(split), stride=32, model_path=MODEL_PATH)
        np.save(os.path.join(OUTPUT_DIR, 'prob_raster_{}'.format(split)), prob.transpose(1,2,0))
