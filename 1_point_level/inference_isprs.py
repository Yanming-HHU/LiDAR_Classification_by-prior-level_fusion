'''Inference the result of ISPRS data

Xiaoqiang Liu  2019/10/13
'''
import copy
from datetime import datetime
import importlib
import logging
import math
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
from sklearn.neighbors import KDTree
from isprs_dataset import ISPRSDataset
import argparse
import pprint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)  #model path
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'common/utils'))   # pointnet2 utils and tf wrapper for pointnet2

import provider
import tf_util
import pc_util

def parse_args(argv):
    # Setup arguments & parse
    parser = argparse.ArgumentParser(
        description=__doc__, # printed with -h/--help
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--model', default='pointnet2_sem_seg', help='Model name [default: pointnet2_sem_seg]')
    parser.add_argument('--extra-dims', type=int, default=[], nargs='*', help='Extra dims')
    parser.add_argument('--model_path', default='data/results/scannet/log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
    parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during inference [default: 16]')
    parser.add_argument('--epoch', type=int, default=3, help='Epoch during inference')
    parser.add_argument('--input_path', default='../data/Vaihingen3D_EVAL_WITH_REF.txt', help='Input point clouds path')
    parser.add_argument('--output_path', default='./inference/', help='Output path')
    
    return parser.parse_args(argv[1:])

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

def get_batch(data, idx, start_idx, end_idx, num_point=8192):
    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, num_point, len(data.columns)))
    for i in range(bsize):
        if start_idx+i < len(data):
            ps, _, _ = data[idx[start_idx+i]]
            batch_data[i,...] = ps
    return batch_data

def inference(data, model, model_path, batch_size, num_point=8192, num_classes=8, gpu_index=0, epoch=3):
    '''inference the result of data using model
    Args:
        data: list, each of list is [npoint, 3]
        model: python module
        model_path: str, the path to model parameter
        batch_size: int, this is for placeholder of tensorflow
        num_point: int, the sampled number of point cloud dataset, this is for placeholder of tensorflow
        num_classes: int, this is for placeholder of tensorflow
    Return:
        all_points: list, the lenth is equal to data. and each of list is the data sampled from data 
                      [8192, 3]
        all_probs: list, the lenth is equal to data. and each of list is the probalibity for labeling data 
              [8192, num_classes]
    '''
    
    with tf.Graph().as_default():
        with tf.device('/device:GPU:'+str(gpu_index)):
            pointclouds_pl, labels_pl, smpws_pl = model.placeholder_inputs(batch_size, num_point)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            logging.info("Loading model")
            pred, end_points = model.get_model(pointclouds_pl, is_training_pl, num_classes)
            saver = tf.train.Saver()
        
        # create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        
        # restore variables from disks
        saver.restore(sess, model_path)
        ops = {'pointclouds_pl': pointclouds_pl,
               'is_training_pl': is_training_pl,
               'pred': pred}
        is_training = False
        logging.info("Model loaded")
        
        num_batches = int(math.ceil((1.0*len(data))/batch_size))
        for e in range(epoch):
            # Shuffle inference samples
            idx = np.arange(0, len(data))
            np.random.shuffle(idx)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx*batch_size
                end_idx = (batch_idx+1)*batch_size
                batch_data = get_batch(data, idx, start_idx, end_idx)
                
                aug_data = np.concatenate((provider.rotate_point_cloud_z(batch_data[:,:,0:3]),
                                           batch_data[:,:,3:]), axis=2)
                feed_dict = {ops['pointclouds_pl']: aug_data,
                             ops['is_training_pl']: is_training}
                pred_val = sess.run([ops['pred']], feed_dict=feed_dict)
                
                # calculating softmax probabilty
                tmp_pred = pred_val[0] - np.expand_dims(np.max(pred_val[0], axis=2), 2)
                tmp_prob = np.exp(tmp_pred)
                pred_prob = tmp_prob / np.expand_dims(np.sum(tmp_prob, axis=2), 2)
                
                if batch_idx == num_batches-1:   # for the last batch, some additional zero vector is faded
                    pred_prob = pred_prob[0:(len(data)-start_idx), :]
                    batch_data = batch_data[:(len(data)-start_idx), :]
                
                pred_prob.shape = (pred_prob.shape[0]*pred_prob.shape[1], pred_prob.shape[2])
                batch_data.shape = (batch_data.shape[0]*batch_data.shape[1], batch_data.shape[2])
                
                if e==0 and batch_idx == 0:
                    all_probs = [pred_prob]
                    all_points = [batch_data]
                else:
                    all_probs += [pred_prob]
                    all_points += [batch_data]
                
    return all_points, all_probs

def predict_for_whole(whole_data, all_points, all_probs, epoch=3):
    '''predict the whole_data using knn for all_probs
    Args:
        whole_data: np.array([n, 3]), the whole point clouds
        all_points: list, each is array(8192,3), including point derived from splitting whole_data
        all_probs:  list, each is array(8192,3), meaning probability for all points
    Return:
        pred_probs: the predicted probability for whole data
    '''
    
    all_points = np.concatenate(all_points, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    
    logging.info('predict for the whole dataset')
    kdt = KDTree(all_points[:, 0:3], metric='euclidean')
    
    whole_data = whole_data[:, 0:3]
    min_ = np.min(whole_data, axis=0)
    whole_data = whole_data - min_
    ind = kdt.query(whole_data[:,0:3], k=epoch, return_distance=False)
  
    
    print(len(whole_data))
    probs_neighbor = all_probs[ind]
    pred_probs = np.average(probs_neighbor, axis=1)
    return pred_probs
    

if __name__ == '__main__':
    flags = parse_args(sys.argv)
    start_log(flags)
    
    #set the parameter about pointnet2
    epoch_cnt = 0
    batch_size = flags.batch_size
    num_point = flags.num_point
    gpu_index = flags.gpu
    model_path = flags.model_path
    epoch = flags.epoch
    model = importlib.import_module(flags.model)    # import network module
    num_classes = 8
    columns = np.array([0,1,2]+flags.extra_dims)
    num_dimensions = len(columns)
    
    # copy the inference file and model file to inference director
    model_file = os.path.join(BASE_DIR, flags.model+'.py')
    os.system('cp %s %s' % (model_file, flags.output_path))
    os.system('cp '+__file__+' %s' % (flags.output_path))
    
    data = ISPRSDataset('./data', split='test', extra_features=flags.extra_dims)
    all_points, all_probs = inference(data, model, model_path, batch_size, num_point, num_classes, epoch=epoch)
    with open(os.path.join(flags.output_path,'all_points.pickle'), 'wb') as f:
        pickle.dump(all_points, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(flags.output_path,'all_probs.pickle'), 'wb') as f:
        pickle.dump(all_probs, f, pickle.HIGHEST_PROTOCOL)
    
    whole_data = np.loadtxt('../data/Vaihingen3D_EVAL_WITH_REF.txt')
    pred_probs = predict_for_whole(whole_data, all_points, all_probs, epoch)
    np.save(os.path.join(flags.output_path, 'predict.npy'), pred_probs)
    logging.info('Done')
    
    
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
    print(all_metric)
    print(mean_metric)
