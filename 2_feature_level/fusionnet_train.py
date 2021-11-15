"""Fusion for lidar and optical image in feature level

Xiaoqiang Liu    2019/07/27
modified         2020/01/03
"""
import os
import sys
import numpy as np
import argparse
import pickle
import gc
from queue import Queue

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from mlp_pytorch import MLP
from segnet_utils import cal_metric

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='./log/fusionnet_train', help='path to the log file')
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--decay_step', type=int, default=2010)
FLAGS = parser.parse_args()
DECAY_STEP=FLAGS.decay_step

BATCH_SIZE = 16
EPOCH = FLAGS.epoch
DEVICE = torch.device("cuda:{}".format(FLAGS.gpu) if torch.cuda.is_available() else "cpu")

LOG_DIR = FLAGS.log_dir
if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'fusionnet.log'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

class PointRasterDataset(Dataset):
    """Dataset contain Point and raster feature"""
    def __init__(self, data, label, transform=None):
        """
        Args:
            data (np.array[NUM_POINT, NUM_FEATURE]):
            label (string): Path to the label file
            transform (callable, optional): Optinal transform to be applied on a sample
        """
        self.data = np.asarray(data, dtype='float32')
        self.label = np.asarray(label, dtype='int64')
        self.tranform = transform
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # change the data to feature to [N_FEATURES, 8192] and label t0 [8192]
        return torch.from_numpy(self.data[idx]), torch.from_numpy(self.label[idx])

def creat_dataloader(x_path, y_path, batch_size=BATCH_SIZE):
    """"creat DataLoader
    Args:
        x_path: string, path to x (feature)
        y_path: string, path to y (label)
        batch_size: int
    Returns:
        DataLoader: torch
    """
    with open(x_path, 'rb') as f:
        x = pickle.load(f)
    with open(y_path, 'rb') as f:
        y = pickle.load(f)
    # The batch is flatterned and the x become [NUM_POINT, N_FEATURES]
    x = x[:, 3:]     # x, y,z are ignored in training
    x.shape = (-1, 8192, x.shape[-1])
    x = x.transpose(0, 2, 1)
    y.shape = (-1, 8192)
    
    tmp = DataLoader(PointRasterDataset(x, y), 
                     batch_size=batch_size, shuffle=False)
    return tmp

def train_one_epoch(data, net, optimizer, scheduler=None, weights=None,
                    global_step=[0], writer=None, epoch=0):
    
    """
    Args:
        train_data: torch.utils.data.DataLoader(data_set, batch_size)
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
    for batch_idx, (batch_data, batch_gts) in enumerate(data):
        # data, target = Variable(data.cuda()), Variable(target.cuda())
        batch_data, batch_gts = batch_data.to(DEVICE), batch_gts.to(DEVICE)
        optimizer.zero_grad()
        out = net(batch_data)
        loss = F.cross_entropy(out, batch_gts, weight=weights)
        loss.backward()
        optimizer.step()
        global_step[0] = global_step[0] + 1
        pred = torch.argmax(out, dim=1)
        
        total = (batch_gts!=7).cpu().numpy()
        tmp_correct = (pred==batch_gts).cpu().numpy() & total 
        accuracy = tmp_correct.sum() / total.sum()
        
        if writer is not None:
            writer.add_scalar('loss', loss.data, global_step = global_step[0])
            writer.add_scalar('accuracy', accuracy, global_step = global_step[0])
        if scheduler is not None:
            scheduler.step()

def test_one_epoch(data, net, all=False, batch_size=BATCH_SIZE,
                   global_step=None, writer=None):
    """
    Args:
        data: torch.utils.data.DataLoader(data_set, batch_size)
        net: torch.model
        all: bool, return all or not
        batch_size: int
        global_step: list, for update
        writer: SummaryWriter in tensorboard
    Returns:
        accuracy: float
        all_preds: list, prdict label
        all_gts: list, groud truth
    """
    net.eval()
    weight = torch.ones(8)
    weight[-1] = 0
    weight = weight.to(DEVICE)
    for batch_idx, (batch_data, batch_gts) in enumerate(data):
        batch_data, batch_gts = batch_data.to(DEVICE), batch_gts.to(DEVICE)
        with torch.no_grad():
            out = net(batch_data)
            loss = F.cross_entropy(out, batch_gts, weight=weight)
        pred = torch.argmax(out, dim=1)

        total = (batch_gts!=7).cpu().numpy()
        tmp_correct = (pred==batch_gts).cpu().numpy() & total
        tmp_accuracy = tmp_correct.sum() / total.sum()

        if writer is not None:
            writer.add_scalar('loss', loss.data, global_step=global_step[0])
            writer.add_scalar('accuracy', tmp_accuracy, global_step=global_step[0])
        
        if batch_idx == 0:
            all_pred = [pred.cpu().numpy()]
            all_gts = [batch_gts.cpu().numpy()]
        else:
            all_pred += [pred.cpu().numpy()]
            all_gts += [batch_gts.cpu().numpy()]
    
    accuracy = cal_metric(np.concatenate([p.ravel() for p in all_gts]),
                          np.concatenate([p.ravel() for p in all_pred]),
                          log_string=log_string)
    if all:
        return accuracy, all_pred, all_gts, 
    else:
        return accuracy


def train(argv=None):
    # data preparation
    x_path = ['/result_output/3_feature_level/data/ds2010/fusion_feature_train.pickle',
              '/result_output/3_feature_level/data/ds2010/fusion_feature_test.pickle']
    y_path = ['/result_output/3_feature_level/data/ds2010/point_label_train.pickle',
              '/result_output/3_feature_level/data/ds2010/point_label_test.pickle']
    
    weight = np.array([180850, 47605, 135173, 193723, 4614, 152045, 27250, 12616], dtype='float32')
    weight = np.median(weight) / weight
    weight = torch.from_numpy(weight)
    weight[-1] = 0
    weight = weight.to(DEVICE)


    trainloader = creat_dataloader(x_path[0], y_path[0])
    testloader = creat_dataloader(x_path[1], y_path[1])
    
    # Define net and
    net = MLP(in_channels=192, out_channels=8)
    net.to(DEVICE)
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [8, 16], gamma=0.2)
    lr_lambda = lambda step: max(0.7**np.floor(step*16/DECAY_STEP), 0.00001)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200, 300], gamma=0.1)
    
    global_step = [0]
    acc_max = 0
    train_writer = SummaryWriter(log_dir = os.path.join(LOG_DIR, 'train'))
    test_writer =  SummaryWriter(log_dir = os.path.join(LOG_DIR, 'test'))
    input_data, _ = next(iter(trainloader))
    input_data = input_data.to(DEVICE)
    train_writer.add_graph(net, input_data)
    save_epoch = Queue(maxsize=5)

    log_string('train')
    for e in range(EPOCH):
        print('epoch:{}'.format(e))
        train_one_epoch(data=trainloader, net=net, optimizer=optimizer, weights=weight,
                        global_step=global_step, epoch=e, scheduler=scheduler,
                        writer=train_writer)
        
        acc_test = test_one_epoch(data=testloader, net=net, global_step=global_step,
                                  writer=test_writer)
        print('acc_test: {}'.format(acc_test))
        if acc_test > acc_max:
            acc_max = acc_test
            # only save the lastest 5 model
            if save_epoch.qsize() < save_epoch.maxsize:
                save_epoch.put(e)
            else:
                del_epoch = save_epoch.get()
                del_path = os.path.join(LOG_DIR, 'best_mlpfusion_{}'.format(del_epoch))
                os.system('rm {}'.format(del_path))
                save_epoch.put(e)
            best_file_path = os.path.join(LOG_DIR, 'best_mlpfusion_{}'.format(e))
            torch.save(net.state_dict(), best_file_path)
            log_string('the best model has saved in {}'.format(best_file_path))

    log_string('end')

if __name__ == '__main__' :
    os.system('cp '+__file__+' %s' % (LOG_DIR))
    train()
