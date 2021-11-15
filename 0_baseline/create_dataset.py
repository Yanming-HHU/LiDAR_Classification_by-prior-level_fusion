"""Creat train and test data for pointNet using ISPRS dataset
This split is referred to pointCNN data preparation for scannet https://github.com/yangyanli/PointCNN/blob/master/data_conversions/prepare_scannet_seg_data.py

Xiaoqiang Liu        2019/10/12        
"""
import numpy as np
import pickle
def split_data(data, window=(30,30), offset=(0,0), min_point_num=2000):
    """Split a point cloud data into n blocks using window size
    Args:
        data: (npoint, 4), Numpy array. include point x, y, z, label 
        window: tuple, window size for split data
        offset: tuple, block can be overlap by set the offset
        min_point_num: int, aftering split, some block may contain few points, these blocks 
                       should be merged to its neighborhood
    Return:
        dataset_xyz: list, element in this is point in certain block, including x,y,z
        dataset_label: list, the label of dataset_xyz
    """
    data_xyz, data_label = data[:,:3], data[:,-1]
    xyz_min = np.min(data_xyz, axis=0)
    data_xyz -= xyz_min     # set the original point of dataset as minmum point
    
    
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
    dataset_xyz = [data_xyz[e] for e in blocks_point_indices if len(e) > 0]
    dataset_label = [data_label[e] for e in blocks_point_indices if len(e) > 0]
    return dataset_xyz, dataset_label

if __name__ == '__main__':
    data_files = ['../data/Vaihingen3D_training.txt',
                  '../data/Vaihingen3D_EVAL_WITH_REF.txt']
    window = (30, 30)
    min_point_num = 2000
    offsets = [(0,0), (window[0]/2.0, window[0]/2.0)]
    
    targets = ['train', 'test']
    for data_file, target in zip(data_files, targets):
        data = np.loadtxt(data_file)
        data = data[:,[0,1,2,-1]]
        for i, offset in enumerate(offsets):
            if i==0:
                dataset_xyz, dataset_label = split_data(data, window, offset, min_point_num)
            else:
                tmp_xyz, tmp_label = split_data(data, window, offset, min_point_num)
                dataset_xyz += tmp_xyz
                dataset_label += tmp_label
        with open('./isprs_data/isprs_{}_dataset.pickle'.format(target), 'wb') as f:
            pickle.dump(dataset_xyz,f,pickle.HIGHEST_PROTOCOL)
        with open('./isprs_data/isprs_{}_labels.pickle'.format(target), 'wb') as f:
            pickle.dump(dataset_label, f, pickle.HIGHEST_PROTOCOL)
