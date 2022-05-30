# -*- coding: utf-8 -*-
import pickle,time
import json
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from sklearn.neighbors import KDTree
import scipy.sparse as sp

class get_dataset(object):
    def __init__(self,anchors):
        assert type(anchors) == type(dict())
        data = np.array(list(anchors.items()))
        #data.sort(axis=0) #这里的排序会导致anchors的对应关系被破坏。
        self.data=data
    def get(self,dset,n=0.3,seed=0):
        if 0<n<1:
            n = int(len(self.data)*n)
        np.random.seed(seed)
        data = self.data.copy()
        np.random.shuffle(data)
        if dset=='train':
            return data[:n]
        elif dset=='test':
            return data[-n:]
        
def get_train_set(train_pair,batch_size,node_size):
    train = np.repeat(train_pair,batch_size//len(train_pair)+1,axis=0)
    np.random.shuffle(train); train = train[:batch_size]
    while True:
        f = np.random.randint(0,node_size,train.shape)
        train_set = np.concatenate([train,f],axis = -1)
        yield train_set
        
def get_sim(embed, embed2 = None, sim_measure = "euclidean", top_k = 10):
    n_nodes, dim = embed.shape
    if embed2 is None:
        embed2 = embed
    kd_sim = kd_align(embed, embed2, distance_metric = sim_measure, top_k = top_k)
    return kd_sim

def kd_align(emb1, emb2, normalize=False, distance_metric = "euclidean", top_k = 10):
    kd_tree = KDTree(emb2, metric = distance_metric)    
        
    row = np.array([])
    col = np.array([])
    data = np.array([])
    
    dist, ind = kd_tree.query(emb1, k = top_k)

    row = np.array([])
    for i in range(emb1.shape[0]):
        row = np.concatenate((row, np.ones(top_k)*i))
    col = ind.flatten()
    data = np.exp(-dist).flatten()
    sparse_align_matrix = coo_matrix((data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))
    return sparse_align_matrix.tocsr()

def hit_precision(sim_matrix, top_k =10, anchors=None):
    if anchors is None:
        n_nodes = sim_matrix.shape[0]
        nodes = list(range(n_nodes))
    else:
        n_nodes = len(anchors)
        nodes = list(anchors.keys())
    
    score = 0
    for test_x in nodes:
        if anchors is None:
            test_y = test_x
        else:
            test_y = int(anchors[test_x])
        assert sp.issparse(sim_matrix)
      
        row_idx, col_idx, values = sp.find(sim_matrix[test_x])
        sorted_idx = col_idx[values.argsort()][-top_k:][::-1]
        
        h_x = 0
        for pos,idx in enumerate(sorted_idx):
            if idx == test_y:
                hit_x = pos+1
                h_x = (top_k-hit_x+1)/top_k
                break
        score += h_x
    score /= n_nodes 
    return score


def test(test,emb):
    test_x,test_y = [],[]
    for k,v in test:
        test_x.append(emb[k])
        test_y.append(emb[v])
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    sim_matrix = get_sim(test_x,test_y,top_k=10)
    score=[]
    for top_k in [1,3,5,10]:
        score_ = hit_precision(sim_matrix,top_k=top_k)
        score.append(score_)
    print(time.ctime(),'\tScore:',' '.join(['%.2f'% (x*100) for x in score]))
    return score

def splitDataset(data='wd'):
    anchors = dict(json.load(open('../data/{}/anchors.txt'.format(data),'r')))
    print('Number of anchors in {}:\t'.format(data),len(anchors))
    dataset = get_dataset(anchors)
    for n_train in [0.1,0.2,0.3,0.4,0.5]:
        train_pair = dataset.get('train',n=n_train,seed=0).tolist()
        test_pair = dataset.get('test',n=1-n_train,seed=0).tolist()
        json.dump(train_pair,open('../data/{}/train_pairs{}.txt'.format(data,n_train),'w'))
        json.dump(test_pair,open('../data/{}/test_pairs{}.txt'.format(data,1-n_train),'w'))

if __name__ == "__main__":
    pass
    # splitDataset('wd')
    # splitDataset('dblp')
