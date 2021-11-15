"""Creat train and test data for pointNet using ISPRS dataset. 
This data include ALS information (x, y, z) and image information (IR, R, G)
This split is referred to pointCNN data preparation for scannet https://github.com/yangyanli/PointCNN/blob/master/data_conversions/prepare_scannet_seg_data.py

Xiaoqiang Liu        2019/11/12
"""
import numpy as np
import pickle
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'common/my_utils'))
from fusion import get_rowcol

def split_data(data, window=(30,30), offset=(0,0), min_point_num=2000):
    """Split a point cloud data into n blocks using window size
    Args:
        data: (npoint, 4+n), Numpy array. include point x, y, z, n features, label 
        window: tuple, window size for split data
        offset: tuple, block can be overlap by set the offset
        min_point_num: int, aftering split, some block may contain few points, these blocks 
                       should   merged to its neighborhood
    Return:
        dataset_point: list, element in this is point in certain block, including x,y,z and other feature
        dataset_label: list, the label of dataset_xyz
    """
    data_xyz, data_point, data_label = data[:,:3], data[:,:-1], data[:,-1]
    xyz_min = np.min(data_xyz, axis=0)
    data_xyz -= xyz_min     # set the original point of dataset as minmum point
    # data_point -= np.concatenate([xyz_min, [0,0,0]])
    
    xyz_min = np.min(data_xyz, axis=0)
    xyz_max = np.max(data_xyz, axis=0)
    window = window + (2*xyz_max[-1]-2*xyz_min[-1],)    # the height dimensinal is not split
    
    offset = offset + (0,)  # z is not offset
    xyz_min -= offset
    
    xyz_blocks = np.floor((data_xyz-xyz_min)/window).astype(int)
    blocks, point_block_indices, block_point_counts = np.unique(xyz_blocks, return_inverse=True, return_counts=True, axis=0)
    blocks_point_indices = np.split(np.argsort(point_block_indices), np.cumsum(block_point_counts[:-1]))
    
    # merge small blocks into one of their big neighbors
    block_to_block_idx_map = dict()
    for block_idx in range(blocks.shape[0]):
        block = (blocks[block_idx][0], blocks[block_idx][1])
        block_to_block_idx_map[block] = block_idx
    
    nbr_block_offsets = [(-1,0), (0,-1), (1,0),(0,1),(-1,-1),(-1,1),(1,1),(1,-1)]
    block_merge_count = 0
    for block_idx in range(blocks.shape[0]):
        if block_point_counts[block_idx] >= min_point_num:
            continue
        
        block = (blocks[block_idx][0], blocks[block_idx][1])
        for x,y in nbr_block_offsets:
            nbr_block = (block[0]+x, block[1]+y)
            if nbr_block not in block_to_block_idx_map:
                continue
            
            nbr_block_idx = block_to_block_idx_map[nbr_block]
            if block_point_counts[nbr_block_idx] < min_point_num:
                continue
            
            blocks_point_indices[nbr_block_idx] = np.concatenate([blocks_point_indices[nbr_block_idx], blocks_point_indices[block_idx]], axis=-1)
            blocks_point_indices[block_idx] = np.array([], dtype=np.int)
        
            block_point_counts[nbr_block_idx] += block_point_counts[block_idx]
            block_point_counts[block_idx] = 0
            block_merge_count += 1
            break
    dataset_point = [data_point[e] for e in blocks_point_indices if len(e) > 0]
    dataset_label = [data_label[e] for e in blocks_point_indices if len(e) > 0]
    return dataset_point, dataset_label

def feature_from_raster(point, raster, geotransform):
    """extract featreu from raster
    Args:
        point: np.array(n, 2) point cloud with (x, y) coordinate
        raster: np.array(h, w, c) raster with c channels
        geotransform: list
    """
    raster_to_point = np.zeros((point.shape[0], raster.shape[-1]))
    for i in range(len(point)):
        col, row = get_rowcol(point[i], geotransform)
        raster_to_point[i] = raster[row][col]
    
    return raster_to_point

if __name__ == '__main__':
    data_files = ['../data/Vaihingen3D_training.txt',
                  '../data/Vaihingen3D_EVAL_WITH_REF.txt']
    image_files = ['/result_output/5_prior_level/data/prob_raster_train.npy',
                   '/result_output/5_prior_level/data/prob_raster_test.npy']
    
    window = (30, 30)
    min_point_num = 2000
    offsets = [(0,0), (window[0]/2.0, window[0]/2.0)]   #overlapping in 15 meter
    
    targets = ['train', 'test']
    # the point cloud is not shift, so we use original coordinates 
    geotransform = {'train': [496833.57, 0.09, 0.0, 5419599.57, 0.0, -0.09],
                    'test': [497058.3, 0.09, 0.0, 5420010.42, 0.0, -0.09]}
    for data_file, image_file, target in zip(data_files, image_files, targets):
        data = np.loadtxt(data_file)
        raster = np.load(image_file)
        feature = feature_from_raster(data[:,0:2], raster, geotransform=geotransform[target])
        data = np.hstack([data[:, 0:3], feature, data[:,-1].reshape([-1,1])])
        for i, offset in enumerate(offsets):
            if i==0:
                dataset_point, dataset_label = split_data(data, window, offset, min_point_num)
            else:
                tmp_point, tmp_label = split_data(data, window, offset, min_point_num)
                dataset_point += tmp_point
                dataset_label += tmp_label
        with open('./data/isprs_{}_dataset.pickle'.format(target), 'wb') as f:
            pickle.dump(dataset_point,f,pickle.HIGHEST_PROTOCOL)
        with open('./data/isprs_{}_labels.pickle'.format(target), 'wb') as f:
            pickle.dump(dataset_label, f, pickle.HIGHEST_PROTOCOL)
