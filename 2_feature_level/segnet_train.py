"""Using ISPRS data to train segnet, and inference
modified based on https://github.com/nshaud/DeepNetsForEO/blob/master/SegNet_PyTorch_v2.ipynb

Xiaoqiang Liu     2019/07/22
modified          2019/11/20
"""
import argparse
import numpy as np
import sys
import os
from glob import glob
from tqdm import tqdm
from skimage import io
import pickle
import gc
import random
from queue import Queue
try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from segnet import SegNet
from segnet_utils import ISPRS_dataset, cal_metric  # for training
from segnet_utils import convert_from_color       # for data convert
from segnet_utils import grouper, count_sliding_window, sliding_window # for testing data
from segnet_utils import cal_weight

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='./log', help='path to the log file')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--gpu', type=int, default=0)
FLAGS = parser.parse_args()

LOG_DIR = FLAGS.log_dir
DEVICE = torch.device("cuda:{}".format(FLAGS.gpu) if torch.cuda.is_available() else "cpu")

# Parameters
WINDOW_SIZE = (256, 256)   #patch size
STRIDE = 32 # Stride for testing
IN_CHANNELS = 3
BATCH_SIZE = 16
EPOCH = FLAGS.epoch
N_FEATURES = 64


# LABELS = ['imp sur', 'buildings', 'low veg', 'trees', 'car', 'clutter', 'others']
LABELS = ['low veg', 'tree', 'imp sur', 'car', 'building', 'background']
N_CLASSES = len(LABELS)
WEIGHTS = torch.from_numpy(np.array([1, 1, 1, 1, 1, 0], dtype='float32'))   #This is weight for every class to handle imbalanced question
WEIGHTS = WEIGHTS.to(DEVICE)
CACHE = True

MAIN_FOLDER = '../data/VaihingenRaster/'
DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'

OUTLIER_DIR = os.path.join(LOG_DIR, 'outlier') 
if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)
if not os.path.isdir(OUTLIER_DIR):
    os.makedirs(OUTLIER_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'segnet_train.log'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

def train_one_epoch(data, net, optimizer, scheduler=None, weights=WEIGHTS,
                    global_step=[0], writer=None, epoch=0):
    
    """
    Args:
        data: torch.utils.data.DataLoader(data_set, batch_size)
        net: torch.model
        optimizer: torch.optimizer
        scheduler: torch.scheduler
        weights: weights for class
        global_step: list(1) using list because this changes in function will also change source
        writter: SummaryWriter
    Returns:
        losses: list, traing loss
        accuracies: list, training accuracy
    """
    
    net.train()
    
    for batch_idx, (data, target) in enumerate(data):
        # data, target = Variable(data.cuda()), Variable(target.cuda())
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output, _ = net(data)
        loss = F.cross_entropy(output, target, weight=weights)
        loss.backward()
        optimizer.step()
        global_step[0] = global_step[0] + 1
        pred = torch.argmax(output, dim=1)
        size = 1.0 * target.shape[0] * target.shape[1] * target.shape[2]
        accuracy = torch.sum(pred==target).cpu().numpy() / size
        if writer is not None:
            writer.add_scalar('loss', loss.data, global_step = global_step[0])
            writer.add_scalar('accuracy', accuracy, global_step = global_step[0])

    if scheduler is not None:
        scheduler.step()

def test_one_epoch(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, 
                   weights=WEIGHTS, window_size=WINDOW_SIZE, global_step=None, writer=None):
    """
    Args:
        net: torch.model
        test_ids: list, index for test
        all: bool, return all or not
        stride: int,
        batch_size: int
        window_size: tuple
        global_step: list, for update
        writer: SummaryWriter in tensorboard
    Returns:
        accuracy: float
        all_preds: list, prdict label
        all_gts: lsit, groud truth
    """
    
    # Use the network on the test set
    test_images = (1.0/255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(convert_from_color(io.imread(LABEL_FOLDER.format(id))), dtype='int64') 
        for id in test_ids)
    
    all_pred = []
    all_gt = []
    all_loss = []
    
    # Switch the network to inference
    net.eval()
    
    for img, gt in tqdm(zip(test_images, test_labels), total=len(test_ids), leave=False):
        pred = np.zeros(img.shape[:2] + (N_CLASSES,))
        
        total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
        all_patches = sliding_window(img, step=stride, window_size=window_size)
        
        for i, coords in enumerate(grouper(batch_size, all_patches)):
            # Build the tensor
            image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
            image_patches = np.asarray(image_patches)
            image_patches = torch.from_numpy(image_patches).to(DEVICE)
            gts_patches = [np.copy(gt[x:x+w, y:y+h]) for x,y,w,h in coords]
            gts_patches = np.asarray(gts_patches)
            gts_patches = torch.from_numpy(gts_patches).to(DEVICE)           
            
            with torch.no_grad():
                outs, _ = net(image_patches)
                loss = F.cross_entropy(outs, gts_patches, weight=weights)
                all_loss.append(loss.data)
                tmp_pred = torch.argmax(outs, dim=1)
                size = 1.0 * gts_patches.shape[0] * gts_patches.shape[1] * gts_patches.shape[2]
                tmp_accuracy = torch.sum(tmp_pred==gts_patches).cpu().numpy() / size
                if writer is not None:
                    writer.add_scalar('loss', loss.data, global_step=global_step[0])
                    writer.add_scalar('accuracy', tmp_accuracy, global_step=global_step[0])
            
            loss_value = int(loss.cpu().numpy()/1.0)
            #if loss_value > 10:
            #    np.save('{}/{}_batch{}_{}.npy'.format(OUTLIER_DIR, 'data', i, loss_value), image_patches.cpu().numpy())
            #    np.save('{}/{}_batch{}_{}.npy'.format(OUTLIER_DIR, 'outs', i, loss_value), outs.cpu().numpy())
            #    np.save('{}/{}_batch{}_{}.npy'.format(OUTLIER_DIR, 'gts', i, loss_value), gts_patches.cpu().numpy())
                
            #Fill in the results array
            outs = outs.data.cpu().numpy()
            for out, (x,y,w,h) in zip(outs, coords):
                out = out.transpose((1,2,0))
                pred[x:x+w, y:y+h] += out

        pred = np.argmax(pred, axis=-1)
        
        all_pred.append(pred)
        all_gt.append(gt)

    accuracy = cal_metric(np.concatenate([p.ravel() for p in all_gt]).ravel(),
                          np.concatenate([p.ravel() for p in all_pred]),
                          log_string=log_string)
    # print("global_step{}".format(global_step[0]))
    # log_string("loss:{}".format(str(all_loss)))
    # if writer is not None:
    #     writer.add_scalar('accuracy', accuracy, global_step=global_step[0])
    #     writer.add_scalar('loss', sum(all_loss)/len(all_loss), global_step=global_step[0])
    if all:
        return accuracy, all_pred, all_gt
    else:
        return accuracy

def train(argv=None):

    net = SegNet(in_channels=IN_CHANNELS, out_channels=N_CLASSES)
    
    # Map VGG16 parameter for segnet
    vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
    if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
        weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')
    
    vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
    mapped_weights = {}
    k_segnets = list(net.state_dict().keys())
    
    for k_segnet in k_segnets:
        #torch 1.x batch normalization paremeter, 0.x doesn't have this parameter
        if 'num_batches_tracked' in k_segnet or '_D' in k_segnet:
            k_segnets.remove(k_segnet)

    for k_vgg, k_segnet in zip(vgg16_weights.keys(), k_segnets):
        if 'feature' in k_vgg:
            mapped_weights[k_segnet] = vgg16_weights[k_vgg]
    
    net.load_state_dict(mapped_weights, strict=False)
    log_string('Loaded VGG-16 weights in SegNet!')
    net.to(DEVICE)

    # split data for training and testing dataset
    file_ids = [1, 3, 5, 7, 11, 13, 15, 17, 21, 23, 26, 28, 30, 32, 34, 37]
    random.shuffle(file_ids)
    train_ids = file_ids[:-4]
    test_ids = file_ids[-4:]
    weight = cal_weight(train_ids)
    weight = weight.to(DEVICE)
    log_string("train_ids:{}".format(str(train_ids)))
    log_string("test_ids:{}".format(str(test_ids)))
    log_string("weight:{}".format(str(weight)))

    train_set = ISPRS_dataset(train_ids, cache=CACHE)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)

    #Designing the global set for segent 
    base_lr = 0.01
    params_dict = dict(net.named_parameters())
    params = []
    for key, value in params_dict.items():   
        # using different learning reate for decode and encode layer
        if '_D' in key:
            params += [{'params':[value], 'lr': base_lr}]
        else:
            params += [{'params':[value], 'lr': base_lr/2}]
    # optimizer = optim.Adam(params, lr=base_lr)
    optimizer = optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=0.0005)
    # optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [35, 55], gamma=0.1)
    global_step = [0]  # using list because list changed in function also change the data in there
    acc_max = 0
    train_writer = SummaryWriter(log_dir = os.path.join(LOG_DIR, 'train'))
    test_writer =  SummaryWriter(log_dir = os.path.join(LOG_DIR, 'test'))
    # input_to_model, _ = next(iter(train_loader))
    input_to_model = torch.rand(16,3,256,256)
    input_to_model = input_to_model.to(DEVICE)
    train_writer.add_graph(net, input_to_model)
    
    save_epoch = Queue(maxsize=5)
    for e in range(EPOCH):
        log_string('epoch: {}'.format(e))
        
        train_one_epoch(data=train_loader, net=net, optimizer=optimizer, 
                        scheduler=scheduler, global_step=global_step, 
                        weights=weight, writer=train_writer, epoch=e)
        
        # Validate with the targest possible stride for faster computing
        acc_test = test_one_epoch(net, test_ids, stride=256, weights=weight,
                                  global_step=global_step, writer=test_writer)
        
        if acc_max < acc_test:
            acc_max = acc_test
            best_file_path = os.path.join(LOG_DIR, 'best_segnet_{}'.format(e))
            torch.save(net.state_dict(), best_file_path)
            log_string('the best model has saved in {}'.format(best_file_path))
            if save_epoch.qsize() < save_epoch.maxsize:
                save_epoch.put(e)
            else:
                del_file_path = os.path.join(LOG_DIR, 'best_segnet_{}'.format(save_epoch.get()))
                os.system('rm {}'.format(del_file_path))
                save_epoch.put(e)
        
    log_string('end')

if __name__ == "__main__":
    log_string('train:')
    os.system('cp '+__file__+' %s' % (LOG_DIR)) # bkp of train procedure
    os.system('cp segnet_utils.py %s' % (LOG_DIR))
    train()
