"""Fusion for lidar and optical image in feature level

Xiaoqiang Liu    2019/07/27
modified         2019/09/14
modified         2019/11/30
"""
import os
import sys
import numpy as np
import argparse
import pickle
import gc
import logging
import pprint
from sklearn.neighbors import KDTree
from sklearn.metrics import precision_recall_fscore_support

import torch
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from mlp_pytorch import MLP

def parse_args(argv):
    # Setup argument & parse
    parser = argparse.ArgumentParser(
        description=__doc__, # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model_path', default='./log/fusion_train/mlp_final')
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--output_path', default='./inference/', help='path to the output')
    
    return parser.parse_args()

def start_log(opts):
    if not os.path.exists(opts.output_path):
        os.makedirs(opts.output_path)

    rootLogger = logging.getLogger()

    logFormatter = logging.Formatter("%(asctime)s %(threadName)s[%(levelname)-3.3s] %(message)s")
    fileHandler = logging.FileHandler(os.path.join(opts.output_path,os.path.splitext(os.path.basename(__file__))[0]+'.log'),mode='w')
    fileHandler.setFormatter(logFormatter)
    fileHandler.setLevel(logging.DEBUG)
    rootLogger.addHandler(fileHandler)

    logFormatter = logging.Formatter("%(threadName)s[%(levelname)-3.3s] %(message)s")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    consoleHandler.setLevel(logging.INFO)
    rootLogger.addHandler(consoleHandler)

    rootLogger.level=logging.DEBUG

    logging.debug('Options:\n'+pprint.pformat(opts.__dict__))

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32768

class InfenceDataset(Dataset):
    """Dataset contain Point and raster feature"""
    def __init__(self, xyz, feature, transform=None):
        """
        Args:
            data (np.array[NUM_POINT, NUM_FEATURE]):
            label (string): Path to the label file
            transform (callable, optional): Optinal transform to be applied on a sample
        """
        self.xyz = np.asarray(xyz, dtype='float32')
        self.feature = np.asarray(feature, dtype='float32')
        self.tranform = transform
    def __len__(self):
        return len(self.xyz)
    
    def __getitem__(self, idx):
        # change the data to feature to [N_FEATURES, 1] and label t0 [1, 1]
        return torch.from_numpy(self.xyz[idx].reshape([-1, 1])), torch.from_numpy(self.feature[idx].reshape([-1, 1]))

def infence_dataloader(x_path, batch_size=BATCH_SIZE):
    '''creat infence DataLoader
    Args:
        x_path: string, path to feature (include x,y,z and n_feature)
        batch_size: int
    Returns:
        DataLoader: torch
    '''
    with open(x_path, 'rb') as f:
        x = pickle.load(f)
    x = x.reshape([-1, x.shape[-1]])
    xyz = x[:, :3]
    feature = x[:, 3:]
    tmp = DataLoader(InfenceDataset(xyz, feature),
                     batch_size=batch_size, shuffle=False)
    return tmp


def inference(data, net, model_path):
    """
    Args:
        data: np.array, num * n_feature
        net: torch.model
        model_path: str, the path to model parameter
    Return:
        all_pros: np.array, um * n_class
    """
    
    net.load_state_dict(torch.load(model_path))
    net.to(DEVICE)
    net.eval()
    for batch_idx, (xyz, feature) in enumerate(data):
        feature = feature.to(DEVICE)
        with torch.no_grad():
            out = net(feature)
        out = torch.exp(out)
        out = out.cpu().numpy()
        xyz = xyz.cpu().numpy()
        xyz.shape = [xyz.shape[0], xyz.shape[1]]
        out.shape = [out.shape[0], out.shape[1]]
        if batch_idx == 0:
            all_prob = [out]
            all_point = [xyz]
        else:
            all_prob += [out]
            all_point += [xyz]
    return np.concatenate(all_point, axis=0), np.concatenate(all_prob, axis=0)

def predict_for_whole(whole_data, all_points, all_probs, epoch=3):
    '''predict the whole_data using knn for all_probs
    Args:
        whole_data: np.array([n, 3]), the whole point clouds
        all_points: list, each is array(8192,3), including point derived from splitting whole_data
        all_probs:  list, each is array(8192,3), meaning probability for all points
    Return:
        pred_probs: the predicted probability for whole data
    '''
    kdt = KDTree(all_points[:, 0:3], metric='euclidean')

    whole_data = whole_data[:, 0:3]
    min_ = np.min(whole_data, axis=0)
    whole_data = whole_data - min_
    ind = kdt.query(whole_data[:,0:3], k=epoch, return_distance=False)

    probs_neighbor = all_probs[ind]
    pred_probs = np.average(probs_neighbor, axis=1)
    return pred_probs


if __name__ == '__main__':
    flags = parse_args(sys.argv)
    start_log(flags)

    epoch = flags.epoch

    logging.info('inference for all test data:')
    data_path = '/result_output/3_feature_level/data/ds2010/fusion_feature_test.pickle'
    dataloader = infence_dataloader(data_path)
        
    net = MLP(in_channels=192, out_channels=8)
    all_points, all_probs = inference(dataloader, net, flags.model_path)

    print('all_point: {}'.format(all_points.shape))
    print('all_probs: {}'.format(all_probs.shape))
    logging.info('inference for ISPRS test data:')
    whole_data = np.loadtxt('../data/Vaihingen3D_EVAL_WITH_REF.txt')
    pred_probs = predict_for_whole(whole_data, all_points, all_probs, epoch)
    np.save(os.path.join(flags.output_path, 'predict.npy'), pred_probs)
    
    #Evaluate result
    from sklearn.metrics import precision_recall_fscore_support
    LABEL_MAP = {0:7, 1:0, 2:3, 3:4, 4:7, 5:5, 6:6, 7:1, 8:2}
    gts = whole_data[:,-1]
    for i in range(len(gts)):
        gts[i] = LABEL_MAP[gts[i]]
    all_metric = precision_recall_fscore_support(gts, np.argmax(pred_probs, axis=-1))
    mean_metric = precision_recall_fscore_support(gts, np.argmax(pred_probs, axis=-1), average='weighted')
    with open(os.path.join(flags.output_path, 'evaluation.pickle'), 'wb') as f:
        pickle.dump(all_metric, f)
        pickle.dump(mean_metric, f)
    
    logging.info('all_metric: {}'.format(str(all_metric)))
    logging.info('mean_metric: {}'.format(str(mean_metric)))
    logging.info('Done')
