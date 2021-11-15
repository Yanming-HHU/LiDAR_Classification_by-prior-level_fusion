import os
import pickle
import sys
import numpy as np
LABEL_MAP = {0 : 7, # Powerline, 546
             1 : 0, # Low vegetation, 180850,
             2 : 3, # Impervious surface, 193723
             3 : 4, # Car, 4614
             4 : 7, # Fence, 12070
             5 : 5, # Roof, 152045
             6 : 6, # Facade, 27250
             7 : 1, # Shrub, 47605
             8 : 2, # Tree, 135173
             }

# The weight is calculate using median frequecy balancing
LABEL_WEIGHT = np.array([180850, 47605, 135173, 193723, 4614, 152045, 27250, 12616])
LABEL_WEIGHT = np.median(LABEL_WEIGHT) / LABEL_WEIGHT
LABEL_WEIGHT[-1] = 0

class ISPRSDataset():
    def __init__(self, root, npoints=8192, split='train', log_weighting=False, extra_features=[]):
        self.npoints = npoints
        self.root = root
        self.split = split
        
        # Dataset size causes memory issues with numpy.save; used pickle instead
        #self.data = np.load(os.path.join(self.root, 'dfc_{}_dataset.npy'.format(split)))
        with open(os.path.join(self.root, 'isprs_{}_dataset.pickle'.format(split)),'rb') as f:
            self.data = pickle.load(f)
        with open(os.path.join(self.root, 'isprs_{}_labels.pickle'.format(split)),'rb') as f:
            self.labels = pickle.load(f)
        
        self.log_weighting = log_weighting
        self.extra_features = extra_features
        self.columns = np.array([0,1,2]+extra_features)
        
        self.M = 8
        
        self.compressed_label_map = LABEL_MAP
        self.labelweights = np.ones(self.M, dtype='float32')

        if split=='train':
            self.labelweights = np.asarray(LABEL_WEIGHT, dtype='float32')
        else:
            self.labelweights = np.ones(self.M, dtype='float32')
        self.labelweights[-1] = 0

    def __getitem__(self, index):
        point_set = self.data[index]
        labels = self.labels[index]
        n = point_set.shape[0]
        
        if self.npoints < n:
            ixs = np.random.choice(n,self.npoints,replace=False)
        elif self.npoints == n:
            ixs = np.arange(self.npoints)
        else:
            ixs = np.random.choice(n,self.npoints,replace=True)
        
        point_set = point_set[ixs,:]
        # point_set = tmp[:,self.columns] / self.scale[self.columns]
        semantic_seg = np.zeros(self.npoints, dtype='int32')
        for i in range(self.npoints):
            semantic_seg[i] = self.compressed_label_map[labels[ixs[i]]]
        sample_weight = self.labelweights[semantic_seg]
        
        return point_set, semantic_seg, sample_weight

    def __len__(self):
        return len(self.data)

if __name__=='__main__':
    d = ISPRSDataset(root = './data', extra_features=[])
    point_set, semantic_seg, sample_weight = d[0]
    print(point_set)
    print(semantic_seg)
    print(sample_weight)
    # print("Scale:"+str(d.scale))
    # print("Mapping: "+str(d.decompress_label_map))
    print("Weights: "+str(d.labelweights))
    # print("Counts: "+str(d.cls_hist))
    # tmp = np.array(list(d.cls_hist.values()),dtype=float)
    # print("Frequency: "+str(tmp/np.sum(tmp)))
    # print("Length: "+str(len(d.data)))
    exit()


