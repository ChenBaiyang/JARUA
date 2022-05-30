#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os,json,pickle,time
import numpy as np
import networkx as nx
from utils import test
from model import JARUA,subword_encode,word_encode
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True  
sess = tf.Session(config=config)  

# In[2]:
def network_extension(g1,g2,train_pair):
    left,right = train_pair[:,0],train_pair[:,1]
    g1_sub = nx.subgraph(g1,left)
    g2_sub = nx.subgraph(g2,right)

    g1_g2_mapping = dict(zip(left,right))
    g2_g1_mapping = dict(zip(right,left))

    g1_sub_enhance = nx.relabel_nodes(g2_sub,g2_g1_mapping,copy=True)
    g2_sub_enhance = nx.relabel_nodes(g1_sub,g1_g2_mapping,copy=True)            
    
    print(time.ctime(),'\t# of edges in network 1/2: \t',g1.size(),g2.size())
    g1.update(g1_sub_enhance)
    g2.update(g2_sub_enhance)
    print(time.ctime(),'\t# of edges in network 1/2: \t',g1.size(),g2.size())
    g1.update(g2)

    return g1

def get_pairs(vec,anchors,users_g1,users_g2,alpha=0.9):
    new_pair = set()
    
    re_users_g1 = sorted(list(users_g1 - set(anchors[:,0])))
    re_users_g2 = sorted(list(users_g2 - set(anchors[:,1])))
    #print(len(re_users_g1),len(re_users_g2))
    
    Lvec = np.array([vec[i] for i in re_users_g1])
    Rvec = np.array([vec[i] for i in re_users_g2])
    
    Lvec = Lvec / np.linalg.norm(Lvec,axis=-1,keepdims=True)
    Rvec = Rvec / np.linalg.norm(Rvec,axis=-1,keepdims=True)
    sim_o = -Lvec.dot(Rvec.T)
    Lsim = sim_o.argsort(-1)
    Rsim = sim_o.argsort(0)
    #print(len(Lsim[:,0]),len(Rsim[0,:]))
    c1,c2=0,0
    for i,j in enumerate(Lsim[:,0]):
        dist = -sim_o[i,j]
        if Rsim[0,j]==i:
            c1+=1
            if dist > alpha:
                c2+=1
                u1,u2 = re_users_g1[i],re_users_g2[j]
                pair = str([u1,u2])
                new_pair.add(pair)
    print('\tPotential pairs:',len(re_users_g1),c1,c2,'\tThreshold:',alpha)
    return new_pair

# In[3]:
result=[]
data_input = int(input('Choose a dataset ( 0 - DBLP, 1 - WD): '))
n_train = float(input('Train set proportion (0.1 - 0.5): '))
data = ['dblp','wd'][data_input]

record_file_name = 'result_JARUA_{}_Ntrain{}.txt'.format(data,n_train)

print(time.ctime(),'\tLoading data...')
G1,G2 = pickle.load(open('../data/{}/networks'.format(data),'rb'))
print(time.ctime(),'\tSize of two networks:',len(G1),len(G2))
attrs = pickle.load(open('../data/{}/attrs'.format(data),'rb'))
users_g1,users_g2 = set(G1.nodes()),set(G2.nodes())
dropout = [[0.5,0.3],[0.3,0.3]][data_input]
batch_size = len(attrs)

subword_corpus, word_corpus = [], []
for i in range(len(attrs)):
    v = attrs[i]
    subword_corpus.append(v[0])
    word_corpus.append(v[1]+v[2]) 
    # The index number is the node id of users in the two networks.

dim = 100
print(time.ctime(),'\tSub-word level attributes embedding...')
subword_feature = subword_encode(subword_corpus, data, dim=dim)

print(time.ctime(),'\tWord level attributes embedding...')
word_feature = word_encode(word_corpus,data,dim=dim)

result = []
for seed in range(2):
    train_pair = np.array(json.load(open('../data/{}/train_pairs{}.txt'.format(data,n_train),'r')))
    test_pair = np.array(json.load(open('../data/{}/test_pairs{}.txt'.format(data,1-n_train),'r')))
    
    score_list = []
    for n_iter in range(10):
        print('-----------------------------------------------------------------------------------------------')    
        print(time.ctime(),'\tSeed:%d, Dataset:%s, n_Iter:%d, n_train:%.2f - %d'%(seed,data,n_iter,n_train,len(train_pair)))
        g1,g2 = G1.copy(),G2.copy()
        G = network_extension(g1,g2,train_pair)
        
        K.clear_session()
        embeddings = JARUA(subword_feature, word_feature, G, train_pair,
                           dim=dim, lr=0.005, gamma=3, k=2,
                           seed=seed, dropout=dropout)

        score = test(test_pair,embeddings)
        record = ['NotFinal',data,seed,n_train,n_iter]+score
        result.append(record)
        print(record)

        # Early stop
        score_list.append(score[-3])
        if n_iter>=2 and score_list[-1] - score_list[-2] < 0.001:
            final = result[-2].copy()
            final[0] = 'Final'
            result.append(final)
            print('Early stop...')
            break
    
        pairs = get_pairs(embeddings,train_pair,users_g1,users_g2,alpha=0.8)
        new_pair = np.array([eval(i) for i in pairs])
        train_pair = np.vstack((train_pair,new_pair))
        print(time.ctime(),'\tFound %d new pairs in Iter - %d'%(len(new_pair),n_iter))  

    break # Uncomment this line for reproductive performance.
    
json.dump(eval(str(result)),open(record_file_name,'a'))




