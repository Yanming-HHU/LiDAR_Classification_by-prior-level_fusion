"""Create fusion data
Xiaoqiang Liu     2019/11/22
"""
import argparse
import glob
import logging
import multiprocessing
import numpy as np
import os
import pickle
import pprint
import random
import sys
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'common/my_utils'))
from fusion import get_rowcol

def parse_args(argv):
    # Setup arguments & parse
    parser = argparse.ArgumentParser(
        description=__doc__, # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument('-i','--input-path', help='e.g. /path/to/DFC/Track4', required=True)
    parser.add_argument('-o','--output-path', default='/result_output/3_feature_level/data',
                        help='e.g. /path/to/training_data_folder',)
    return parser.parse_args(argv[1:])

def start_log(opts):
    if not os.path.exists(opts.output_path):
        os.makedirs(opts.output_path)

    rootLogger = logging.getLogger()

    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-3.3s] %(message)s")
    fileHandler = logging.FileHandler(os.path.join(opts.output_path,os.path.splitext(os.path.basename(__file__))[0]+'.log'),mode='w')
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.DEBUG)
    rootLogger.addHandler(fileHandler)

    logFormatter = logging.Formatter("[%(levelname)-3.3s] %(message)s")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    consoleHandler.setLevel(logging.INFO)
    rootLogger.addHandler(consoleHandler)

    rootLogger.level=logging.DEBUG

    logging.debug('Options:\n'+pprint.pformat(opts.__dict__))

def feature_fusion(point_path, raster_path, geotransform):
    """feature fusion
    Args:
        point_path: .pickle file, path to point cloud feature
        raster_path: .npy file, path to raster feature
        geotransform: tuple, transform parameters between point cloud and raster
    Returns:
        fusion_feature: np.array, fusion feature
    """
    if not os.path.isfile(point_path):
        raise IOError('No File: {}'.format(point_path))
    if not os.path.isfile(raster_path):
        raise IOError('No File: {}'.format(raster_path))
    
    with open(point_path, 'rb') as f:
        point_feature = pickle.load(f)
    raster_feature = np.load(raster_path)
    print("raster_feature:{}".format(raster_feature.shape))
    print("point_feature:{}".format(point_feature.shape))
    raster_to_point = np.zeros((point_feature.shape[0],raster_feature.shape[-1]))
    for i in tqdm(range(len(point_feature)), total=len(point_feature), leave=False):
        col, row = get_rowcol(point_feature[i][0:2], geotransform)
        raster_to_point[i] = raster_feature[row][col]
    
    fusion_feature = np.concatenate((point_feature, raster_to_point), axis=-1)
    return fusion_feature


if __name__ == '__main__':

    flags = parse_args(sys.argv)
    start_log(flags)
    # point_cloud is moved to (0,0,0), so the geotransform -= min(point_cloud)
    geotransform = {'train': [-15.04, 0.09, 0.0, 420.36, 0.0, -0.09],
                    'test':  [-15.03, 0.09, 0.0, 417.64, 0.0, -0.09]}
    # create feature for training and testing
    for type in ['train', 'test']:
        logging.info('this is for {}'.format(type))
        point_feature_path = './data/ds2010/point_feature_{}.pickle'.format(type)
        raster_feature_path = '/result_output/3_feature_level/data/raster_feature_{}.npy'.format(type)
        fusion_feature = feature_fusion(point_feature_path, raster_feature_path, geotransform[type])
        with open(os.path.join(flags.output_path, 'fusion_feature_{}.pickle'.format(type)), 'wb') as f:
            pickle.dump(fusion_feature, f, pickle.HIGHEST_PROTOCOL)

    logging.info('Done')
