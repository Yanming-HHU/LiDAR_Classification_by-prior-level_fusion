import argparse
import math
from datetime import datetime
import numpy as np
import tensorflow as tf
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR) # model
# sys.path.append(os.path.join(ROOT_DIR, 'models')) # no, really model
# sys.path.append(os.path.join(ROOT_DIR, 'common')) # provider
sys.path.append(os.path.join(ROOT_DIR, 'common/utils'))

import provider
import tf_util
import pc_util

# sys.path.append(os.path.join(ROOT_DIR, 'data_prep'))
import isprs_dataset
from isprs_dataset import ISPRSDataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_sem_seg', help='Model name [default: pointnet2_sem_seg]')
parser.add_argument('--data_dir', default=os.path.join(os.path.dirname(ROOT_DIR),'data'), help='Path to training dataset directory')
parser.add_argument('--log_dir', default='log/dfc', help='Log dir [default: log/dfc]')
parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
parser.add_argument('--max_epoch', type=int, default=20, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=68720, help='Decay step for lr decay [default: 68720 (=40x number of training point clouds after split)]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--existing-model', default='', help='Path to existing model to continue to train on')
parser.add_argument('--starting-epoch', type=int, default=0, help='Initial epoch number (for use when reloading model')
parser.add_argument('--extra-dims', type=int, default=[3,4,5,6,7,8], nargs='*', help='Extra dims')
parser.add_argument('--log-weighting', dest='log_weighting', action='store_true')
parser.add_argument('--no-log-weighting', dest='log_weighting', action='store_false')
parser.add_argument('--base_model', default='', help='Path to base model to fine tunning')
parser.set_defaults(log_weighting=True)
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py') # model file
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp '+__file__+' %s' % (LOG_DIR)) # bkp of train procedure
# os.system('cp %s %s' % (os.path.join(FLAGS.data_dir,'dfc_train_metadata.pickle'),LOG_DIR)) # copy of training metadata
LOG_FOUT = open(os.path.join(LOG_DIR, 'train.log'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

NUM_CLASSES = 8

# official train/test split
TRAIN_DATASET = ISPRSDataset(root=FLAGS.data_dir, npoints=NUM_POINT, split='train', 
                             log_weighting=FLAGS.log_weighting,extra_features=FLAGS.extra_dims)
TEST_DATASET = ISPRSDataset(root=FLAGS.data_dir, npoints=NUM_POINT, split='test',
                            extra_features=FLAGS.extra_dims)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.compat.v1.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.compat.v1.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/device:GPU:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.compat.v1.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, smpws_pl)
            tf.compat.v1.summary.scalar('loss', loss)

            pred_class = tf.argmax(pred, 2)
            true_class = tf.cast(labels_pl, tf.int64)
            correct = tf.logical_and(tf.equal(pred_class, true_class), smpws_pl>0)
            # accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / tf.reduce_sum(tf.cast(smpws_pl>0, tf.float32))
            
            tf.compat.v1.summary.scalar('accuracy', accuracy)
            for l in range(NUM_CLASSES):
                a = tf.equal(pred_class, tf.cast(l, tf.int64))
                b = tf.equal(true_class, tf.cast(l, tf.int64))
                A = tf.reduce_sum(tf.cast(tf.logical_and(a,b),tf.float32))
                B = tf.reduce_sum(tf.cast(tf.logical_or(a,b),tf.float32))
                precision = tf.divide(A,tf.reduce_sum(tf.cast(a, tf.float32)))
                recall = tf.divide(A,tf.reduce_sum(tf.cast(b,tf.float32)))
                f1_score = tf.divide(tf.multiply(2.0,tf.multiply(precision, recall)), tf.add(1e-16, tf.add(precision, recall)))
                iou = tf.divide(A,B)
                if l == 0:
                    miou = iou
                else:
                    miou += iou
                tf.compat.v1.summary.scalar('iou_{}'.format(l), iou)
                tf.compat.v1.summary.scalar('precision_{}'.format(l),precision)
                tf.compat.v1.summary.scalar('recall_{}'.format(l), recall)
                tf.compat.v1.summary.scalar('f1_score_{}'.format(l), f1_score)
            miou = tf.divide(miou,tf.cast(NUM_CLASSES-1, tf.float32))
            tf.compat.v1.summary.scalar('mIOU', miou)

            # Add ops to save and restore constant variable
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            fc2_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fc2')
            for var in fc2_var_list:
                var_list.remove(var)
            saver_tuning = tf.train.Saver(var_list)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            # train_op = optimizer.minimize(loss, global_step=batch, var_list=fc2_var_list)

            # Add ops to save and restore all the variables.
            saver = tf.compat.v1.train.Saver(max_to_keep=5,keep_checkpoint_every_n_hours=1)
            bestsaver = tf.compat.v1.train.Saver()
        
        # Create a session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.compat.v1.Session(config=config)
        
        if FLAGS.existing_model:
            log_string("Loading model from "+FLAGS.existing_model)
            saver.restore(sess, FLAGS.existing_model)

        # Add summary writers
        merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        if not FLAGS.existing_model:
            init = tf.global_variables_initializer()
            sess.run(init)

        if FLAGS.base_model:
            log_string("Loading model from" + FLAGS.base_model + "for file tuning")
            init = tf.global_variables_initializer()
            sess.run(init)
            saver_tuning.restore(sess, FLAGS.base_model)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'smpws_pl': smpws_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_acc = -1
        for epoch in range(FLAGS.starting_epoch,MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            
            do_save = epoch % 5 == 0
            
            log_string(str(datetime.now()))
            log_string('---- EPOCH %03d EVALUATION ----'%(epoch))
            acc = eval_one_epoch(sess, ops, test_writer)
            if acc > best_acc:
                best_acc = acc
                save_path = bestsaver.save(sess, os.path.join(LOG_DIR, "best_model.ckpt"), global_step=epoch)
                log_string("Model saved in file: %s" % save_path)
                do_save = False

            # Save the variables to disk.
            if do_save:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step=(epoch*len(TRAIN_DATASET)))
                log_string("Model saved in file: %s" % save_path)

def get_batch_wdp(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, len(TRAIN_DATASET.columns)))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        if start_idx+i < len(dataset):
            ps,seg,smpw = dataset[idxs[i+start_idx]]
            batch_data[i,...] = ps
            batch_label[i,:] = seg
            batch_smpw[i,:] = smpw
            
            dropout_ratio = np.random.random()*0.875 # 0-0.875
            drop_idx = np.where(np.random.random((ps.shape[0]))<=dropout_ratio)[0]
            batch_data[i,drop_idx,:] = batch_data[i,0,:]
            batch_label[i,drop_idx] = batch_label[i,0]
            batch_smpw[i,drop_idx] *= 0
    return batch_data, batch_label, batch_smpw

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, len(TRAIN_DATASET.columns)))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        if start_idx+i < len(dataset):
            ps,seg,smpw = dataset[idxs[i+start_idx]]
            batch_data[i,...] = ps
            batch_label[i,:] = seg
            batch_smpw[i,:] = smpw
    return batch_data, batch_label, batch_smpw

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = int(math.ceil((1.0*len(TRAIN_DATASET))/BATCH_SIZE))
    
    log_string(str(datetime.now()))

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data, batch_label, batch_smpw = get_batch_wdp(TRAIN_DATASET, train_idxs, start_idx, end_idx)
        # Augment batched point clouds by rotation
        if FLAGS.extra_dims:
            aug_data = np.concatenate((provider.rotate_point_cloud_z(batch_data[:,:,0:3]),
                    batch_data[:,:,3:]),axis=2)
        else:
            aug_data = provider.rotate_point_cloud_z(batch_data)
        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['labels_pl']: batch_label,
                     ops['smpws_pl']:batch_smpw,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum((pred_val == batch_label) & (batch_smpw>0))
        total_seen = np.sum(batch_smpw>0)
        
        log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
        log_string('mean loss: %f' % (loss_val))
        log_string('accuracy: %f' % (correct / float(total_seen)))

# evaluate on randomly chopped scenes
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = int(math.ceil((1.0*len(TEST_DATASET))/BATCH_SIZE))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    labelweights = np.zeros(NUM_CLASSES)
    tp = np.zeros(NUM_CLASSES)
    fp = np.zeros(NUM_CLASSES)
    fn = np.zeros(NUM_CLASSES)
        
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data, batch_label, batch_smpw = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)

        if FLAGS.extra_dims:
            aug_data = np.concatenate((provider.rotate_point_cloud_z(batch_data[:,:,0:3]),
                    batch_data[:,:,3:]),axis=2)
        else:
            aug_data = provider.rotate_point_cloud_z(batch_data)

        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['labels_pl']: batch_label,
                     ops['smpws_pl']: batch_smpw,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)

        pred_val = np.argmax(pred_val, 2) # BxN
        correct = np.sum((pred_val == batch_label) & (batch_smpw>0)) # evaluate only all categories except unknown
        total_correct += correct
        total_seen += np.sum(batch_smpw>0)
        loss_sum += loss_val
    
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(batch_label==l)
            total_correct_class[l] += np.sum((pred_val==l) & (batch_label==l) & (batch_smpw>0))
            tp[l] += ((pred_val==l) & (batch_label==l)).sum()
            fp[l] += ((pred_val==l) & (batch_label!=l)).sum()
            fn[l] += ((pred_val!=l) & (batch_label==l)).sum()

    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval point accuracy: %f'% (total_correct / float(total_seen)))
    per_class_str = '     '
    iou = np.divide(tp,tp+fp+fn)
    for l in range(NUM_CLASSES):
        # per_class_str += 'class %d[%d] acc: %f, iou: %f; ' % (TEST_DATASET.decompress_label_map[l],l,total_correct_class[l]/float(total_seen_class[l]),iou[l])
        per_class_str += 'class %d[%d] acc: %f, iou: %f; ' % (l,l,total_correct_class[l]/float(total_seen_class[l]),iou[l])
    log_string(per_class_str)
    log_string('mIOU: {}'.format(iou.sum()/(len(iou)-1)))
    
    return total_correct/float(total_seen)


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
