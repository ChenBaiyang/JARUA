# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import json,pickle,os,time,re
from layer import TokenEmbedding,GAT
from utils import get_train_set
from gensim import utils,models
from gensim.corpora import Dictionary
import tensorflow as tf
from sklearn.decomposition import PCA,TruncatedSVD
import scipy.sparse as sp
import keras
from keras import activations, constraints, initializers, regularizers
from keras.initializers import glorot_uniform
from keras import backend as K
from keras.layers import Layer,Dropout,Dense,Lambda,Input,Input,LeakyReLU,BatchNormalization
import jieba, zhconv
from pypinyin import lazy_pinyin
p_tokenizer = re.compile('[^\u4e00-\u9fa5]')

def tokenizer_cn(text):
    text = zhconv.convert(text,'zh-hans').strip() #Standardize to simple Chinese
    text = p_tokenizer.sub('',text)
    return jieba.lcut(text)
        
def preproc_word(docs,min_len=2,max_len=15):
    
    for i in range(len(docs)):
        docs[i] = [token for token in 
                    utils.tokenize(docs[i],
                                   lower=True,
                                   deacc=True,
                                   errors='ignore')
                    if min_len <= len(token) <= max_len]
    
    from nltk.stem.wordnet import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
    # e.g.: years->year, models->model, modeling->modeling
    
    # NLTK Stop words
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    stop_words = set(stop_words)
    
    docs = [[word for word in document if word not in stop_words] for document in docs]
    
    return docs

def preproc_subword(docs, min_len=2, max_len=15):

    punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    docs = [[w for w in list(doc) if 
          not w.isnumeric() 
          and not w.isspace()
          and not w in punc] for doc in docs]

    # Build the bigram and trigram models
    bigram = models.Phrases(docs, min_count=5, threshold=0.1) # higher threshold fewer phrases.
    trigram = models.Phrases(bigram[docs], threshold=0.1)  
    
    # Get a sentence clubbed as a trigram/bigram
    bigram_mod = models.phrases.Phraser(bigram)
    trigram_mod = models.phrases.Phraser(trigram)

    docs = [bigram_mod[doc] for doc in docs]
    docs = [trigram_mod[bigram_mod[doc]] for doc in docs]

    return docs

def word_encode(docs,dataset,dim=50):
    print(time.ctime(),'\t\tPreprocessing word-level attribute text...') 
    
    if dataset == 'wd':
        print(time.ctime(),'\t\tTokenizing Chinese characters...')
        docs = [tokenizer_cn(doc) for doc in docs]
        stop_words = pickle.load(open('../data/wd/stop_words_cn.pkl','rb'))
        stop_words = set(stop_words)
        docs = [[word for word in document if word not in stop_words] for document in docs]
    else:
        docs = preproc_word(docs)
        
    #print(time.ctime(),'\t\tTo counter vector...') 
    dict_ = Dictionary(docs)
    dict_.filter_extremes(no_below=10, no_above=0.5)
    
    docs = [dict_.doc2bow(doc) for doc in docs]
    n_of_t,n_of_d = len(dict_),len(docs)

    print(time.ctime(),'\t\t# of unique words: %d' % n_of_t)
    #print(time.ctime(),'\t\t# of users: %d' % n_of_d)
    data = sp.lil_matrix((n_of_d,n_of_t),dtype=np.int32)
    for n,values in enumerate(docs):
        for idx,value in values:
            data[n,idx] = value

    print(time.ctime(),'\tSVD dimension deduction...')
    svd = TruncatedSVD(n_components=dim, n_iter=10, random_state=0)
    word_feature = svd.fit_transform(data)
    print(svd.explained_variance_ratio_.sum())

    return word_feature

def subword_encode(docs, dataset, dim=50):
    if dataset == 'wd':
        print(time.ctime(),'\t\tChinese characters to pinyin...')
        docs = [''.join(lazy_pinyin(doc)).lower() for doc in docs]
        
    docs = preproc_subword(docs)
    dict_ = Dictionary(docs)
    docs = [dict_.doc2bow(doc) for doc in docs]
    n_of_t,n_of_d = len(dict_),len(docs)
    print(time.ctime(),'\t\t# of unique tokens: %d' % n_of_t)
    #print(time.ctime(),'\t\t# of users: %d' % n_of_d)
    
    data = np.zeros((len(docs),len(dict_)))
    for n,values in enumerate(docs):
        for idx,value in values:
            data[n][idx] = value

    if dim > n_of_t:
        print('Token numbers less than dimension, SVD passed...')
        return data
    
    print(time.ctime(),'\tSVD dimension deduction...')
    svd = TruncatedSVD(n_components=dim, n_iter=10, random_state=0)
    subword_feature = svd.fit_transform(data)
    print(svd.explained_variance_ratio_.sum())

    return subword_feature

def get_JARUA_model(node_size,char_f_size,word_f_size,hidden_dim,batch_size,
                    dropout=[0.5,0.3],gamma=3,lr=0.005,seed=0,k=2,depth=3,activation='relu'):
    # print('Dropout,Gamma,depth,dim,k:',dropout,gamma,depth,hidden_dim,k)
    train_pair = Input(shape=(None,4))
    char_f1 = Input(shape=(None,char_f_size))
    word_f1 = Input(shape=(None,word_f_size))
    
    def squeeze(tensor):
        return K.cast(K.squeeze(tensor,axis=0), dtype='float32')
    
    char_f = Lambda(squeeze)(char_f1)
    word_f = Lambda(squeeze)(word_f1)

    hidden1 = Dense(hidden_dim,activation=activation,
                    kernel_initializer=glorot_uniform(seed=seed))(char_f)
    hidden1 = BatchNormalization()(hidden1)
    output1 = Dense(hidden_dim,activation=activation,
                    kernel_initializer=glorot_uniform(seed=seed))(hidden1)
    output1 = Dropout(dropout[0],seed=seed)(output1)
    
    hidden2 = Dense(hidden_dim,activation=activation,
                    kernel_initializer=glorot_uniform(seed=seed))(word_f)
    hidden2 = BatchNormalization()(hidden2)
    output2 = Dense(hidden_dim,activation=activation,
                    kernel_initializer=glorot_uniform(seed=seed))(hidden2)
    output2 = Dropout(dropout[0],seed=seed)(output2)
    
    adj = Input(shape=(None,2))
    adj_self = Input(shape=(None,3))
    
    node_f = TokenEmbedding(node_size,hidden_dim,trainable = True)(adj) 
    gat_in = [node_f,adj,adj_self]

    output3 = GAT(node_size,activation='relu',
                       depth = depth,
                       attn_heads=k,
                       seed=seed)(gat_in)
    output3 = Dropout(dropout[1])(output3)
    output = Lambda(lambda x:K.concatenate(x,axis=-1))([output1,output2,output3])
    
    find = Lambda(lambda x:K.gather(reference=x[0],indices=K.cast(K.squeeze(x[1],axis=0), 'int32')))([output,train_pair])
    
    def loss_function(tensor):
        def dis(ll,rr):
            return K.sum(K.abs(ll-rr),axis=-1,keepdims=True)
        l,r,fl,fr = [tensor[:,0,:],tensor[:,1,:],tensor[:,2,:],tensor[:,3,:]]
        loss = K.relu(gamma + dis(l,r) - dis(l,fr)) + K.relu(gamma + dis(l,r) - dis(fl,r))
        return tf.reduce_sum(loss,keepdims=True) / (batch_size)
    loss = Lambda(loss_function)(find)
    
    adam_opt = keras.optimizers.adam(lr=lr)
    train_model = keras.Model(inputs = [train_pair,char_f1,word_f1,adj,adj_self],outputs = loss)
    train_model.compile(loss=lambda y_true,y_pred: y_pred,optimizer=adam_opt)

    feature_model = keras.Model(inputs = [char_f1,word_f1,adj,adj_self],outputs = [output])
    return train_model,feature_model

def JARUA(subword_feature, word_feature, G, train_pair,
          dim=100, lr=0.005, seed=0, gamma=3, depth=3,
          k=2, dropout=[0.5,0.3]):
    adj_matrix = nx.adjacency_matrix(G)
    adj_self_matrix = adj_matrix+sp.eye(len(G),dtype='int32')
    adj_self_matrix = adj_self_matrix/adj_self_matrix.sum(axis=1)
    
    adj = np.stack(adj_matrix.nonzero(),axis=1)
    adj_self = sp.find(adj_self_matrix)
    adj_self = np.stack(adj_self,axis=1)
    
    node_size = adj_matrix.shape[0]
    batch_size = node_size

    char_f_size = subword_feature.shape[1]
    word_f_size = word_feature.shape[1] 
    
    print(time.ctime(),'\tTraining the JARUA model...')
    model,get_emb = get_JARUA_model(lr=lr,node_size=node_size,
                              char_f_size=char_f_size, word_f_size=word_f_size,gamma=gamma,hidden_dim=dim,
                              batch_size = batch_size,seed=seed,dropout=dropout,k=k,depth=depth)
    #model.summary()
    losses,ave_loss =[],[]
    batch = get_train_set(train_pair,batch_size,node_size)
    for epoch in range(2000):
        train_set = next(batch)
        inputs = [train_set, subword_feature, word_feature, adj, adj_self]
        inputs = [np.expand_dims(item,axis=0) for item in inputs]

        loss = model.train_on_batch(inputs, np.array([0]))
        losses.append(loss)
        ave_loss.append(np.mean(losses[-100:]))
        
        # Early stop:
        if epoch > 500 and loss<ave_loss[-1] and ave_loss[-1] > ave_loss[-100]:
        	break
    print(time.ctime(),'\tEpoch %d \tloss=%.5f \tave_loss=%.4f'%(epoch+1,loss,ave_loss[-1]))
    vec = get_emb.predict_on_batch(inputs[1:])
    return vec

def get_JARUA_t_model(node_size, f_size, hidden_dim, batch_size,
                      dropout=0.3, gamma=3, lr=0.005, seed=0, k=2, depth=3, activation='relu'):
    print('Dropout,Gamma,depth,dim,k:',dropout,gamma,depth,hidden_dim,k)
    train_pair = Input(shape=(None,4))
    attr_feature = Input(shape=(None,f_size))
    
    def squeeze(tensor):
        return K.cast(K.squeeze(tensor,axis=0), dtype='float32')
    
    attr_f = Lambda(squeeze)(attr_feature)
    output1 = Dense(hidden_dim[0],activation=activation,
                   kernel_initializer=glorot_uniform(seed=seed))(attr_f)
    output1 = Dropout(dropout)(output1)
    
    adj = Input(shape=(None,2))
    adj_self = Input(shape=(None,3))
    
    node_f = TokenEmbedding(node_size,hidden_dim[1],trainable = True)(adj)
    gat_in = [node_f,adj,adj_self]

    output2 = GAT(node_size,activation='relu',
                       depth = depth,
                       attn_heads=k,
                       seed=seed)(gat_in)
    output2 = Dropout(0.3)(output2)
    output = Lambda(lambda x:K.concatenate(x,axis=-1))([output1,output2])
    
    find = Lambda(lambda x:K.gather(reference=x[0],indices=K.cast(K.squeeze(x[1],axis=0), 'int32')))([output,train_pair])
    
    def loss_function(tensor):
        def dis(ll,rr):
            return K.sum(K.abs(ll-rr),axis=-1,keepdims=True)
        l,r,fl,fr = [tensor[:,0,:],tensor[:,1,:],tensor[:,2,:],tensor[:,3,:]]
        loss = K.relu(gamma + dis(l,r) - dis(l,fr)) + K.relu(gamma + dis(l,r) - dis(fl,r))
        return tf.reduce_sum(loss,keepdims=True) / (batch_size)
    loss = Lambda(loss_function)(find)
    
    adam_opt = keras.optimizers.adam(lr=lr)
    train_model = keras.Model(inputs = [train_pair,attr_feature,adj,adj_self],outputs = loss)
    train_model.compile(loss=lambda y_true,y_pred: y_pred,optimizer=adam_opt)

    feature_model = keras.Model(inputs = [attr_feature,adj,adj_self],outputs = [output])
    return train_model,feature_model

def JARUA_t(attr_feature, G, train_pair, dim=[100, 100], lr=0.005, seed=0, gamma=3, depth=3, k=2, dropout=0.3):
    adj_matrix = nx.adjacency_matrix(G)
    adj_self_matrix = adj_matrix+sp.eye(len(G),dtype='int32')
    adj_self_matrix = adj_self_matrix/adj_self_matrix.sum(axis=1)
    
    adj = np.stack(adj_matrix.nonzero(),axis=1)
    adj_self = sp.find(adj_self_matrix)
    adj_self = np.stack(adj_self,axis=1)
    
    node_size = adj_matrix.shape[0]
    batch_size = node_size

    f_size = attr_feature.shape[1]
    
    print(time.ctime(),'\tTraining the JARUA_t model...')
    model,get_emb = get_JARUA_t_model(lr=lr, node_size=node_size,
                                      f_size=f_size, gamma=gamma, hidden_dim=dim,
                                      batch_size = batch_size, seed=seed, dropout=dropout, k=k, depth=depth)
    #model.summary()
    losses,ave_loss =[],[]
    batch = get_train_set(train_pair,batch_size,node_size)
    for epoch in range(2000):
        train_set = next(batch)
        inputs = [train_set,attr_feature,adj,adj_self]
        inputs = [np.expand_dims(item,axis=0) for item in inputs]

        loss = model.train_on_batch(inputs, np.array([0]))
        losses.append(loss)
        ave_loss.append(np.mean(losses[-100:]))
        
        # Early stop:
        if epoch > 500 and loss<ave_loss[-1] and ave_loss[-1] > ave_loss[-100]:
            break
    print(time.ctime(),'\tEpoch %d \tloss=%.5f \tave_loss=%.4f'%(epoch+1,loss,ave_loss[-1]))
    vec = get_emb.predict_on_batch(inputs[1:])
    return vec

def get_JARUA_a_model(char_f_size,word_f_size,hidden_dim,batch_size, dropout_rate=0.3,gamma = 3,lr = 0.005,seed=0,activation='relu'):
    print('Dropout,Gamma,depth,dim,lr:', dropout_rate, gamma, hidden_dim, lr)
    train_pair = Input(shape=(None,4))
    char_f1 = Input(shape=(None,char_f_size))
    word_f1 = Input(shape=(None,word_f_size))
    
    def squeeze(tensor):
        return K.cast(K.squeeze(tensor,axis=0), dtype='float32')

    char_f = Lambda(squeeze)(char_f1)
    word_f = Lambda(squeeze)(word_f1)

    hidden1 = Dense(hidden_dim,activation=activation,
                    kernel_initializer=glorot_uniform(seed=seed))(char_f)
    hidden1 = BatchNormalization()(hidden1)
    output1 = Dense(hidden_dim,activation=activation,
                    kernel_initializer=glorot_uniform(seed=seed))(hidden1)
    output1 = Dropout(dropout_rate,seed=seed)(output1)
    
    hidden2 = Dense(hidden_dim,activation=activation,
                    kernel_initializer=glorot_uniform(seed=seed))(word_f)
    hidden2 = BatchNormalization()(hidden2)
    output2 = Dense(hidden_dim,activation=activation,
                    kernel_initializer=glorot_uniform(seed=seed))(hidden2)
    output2 = Dropout(dropout_rate,seed=seed)(output2)
    
    output = Lambda(lambda x:K.concatenate(x,axis=-1))([output1,output2])
    find = Lambda(lambda x:K.gather(reference=x[0],indices=K.cast(K.squeeze(x[1],axis=0), 'int32')))([output,train_pair])

    
    def loss_function(tensor):
        def dis(ll,rr):
            return K.sum(K.abs(ll-rr),axis=-1,keepdims=True)
        l,r,fl,fr = [tensor[:,0,:],tensor[:,1,:],tensor[:,2,:],tensor[:,3,:]]
        loss = K.relu(gamma + dis(l,r) - dis(l,fr)) + K.relu(gamma + dis(l,r) - dis(fl,r))
        return tf.reduce_sum(loss,keepdims=True) / (batch_size)
    loss = Lambda(loss_function)(find)
    
    adam_opt = keras.optimizers.adam(lr=lr)
    train_model = keras.Model(inputs = [train_pair,char_f1,word_f1],outputs = loss)
    train_model.compile(loss=lambda y_true,y_pred: y_pred,optimizer=adam_opt)

    feature_model = keras.Model(inputs = [char_f1,word_f1],outputs = [output])
    return train_model,feature_model

def JARUA_a(char_feature,word_feature,train_pair,dropout=0.3, dim=100,lr=0.005,seed=0):
    
    node_size,char_f_size = char_feature.shape
    word_f_size = word_feature.shape[1] 
    batch_size = node_size
    
    print(time.ctime(),'\tTraining the attribute-based model...') 
    model,get_emb = get_JARUA_a_model(lr=lr, dropout_rate=dropout,
                              char_f_size=char_f_size, word_f_size=word_f_size,gamma=3,hidden_dim=dim,
                              batch_size = batch_size,seed=seed)
    #model.summary()
    losses,ave_loss =[],[]
    batch = get_train_set(train_pair,batch_size,node_size)
    for epoch in range(2000):
        train_set = next(batch)
        inputs = [train_set,char_feature,word_feature]
        inputs = [np.expand_dims(item,axis=0) for item in inputs]

        loss = model.train_on_batch(inputs, np.array([0]))
        losses.append(loss)
        ave_loss.append(np.mean(losses[-100:]))
        if epoch > 500 and loss<ave_loss[-1] and ave_loss[-1] > ave_loss[-100]:
        	break
    print(time.ctime(),'\tEpoch %d \tloss=%.5f \tave_loss=%.4f'%(epoch+1,loss,ave_loss[-1]))
    vec = get_emb.predict_on_batch(inputs[1:])
    return vec
        
def get_JARUA_s_model(node_size,hidden_dim,batch_size,dropout_rate=0.3,gamma = 3,lr = 0.005,seed=0):

    train_pair = Input(shape=(None,4))
    adj = Input(shape=(None,2))
    adj_self = Input(shape=(None,3))
    
    node_f = TokenEmbedding(node_size,hidden_dim,trainable = True)(adj) 
    gat_in = [node_f,adj,adj_self]

    node_feature = GAT(node_size,activation='relu',
                       depth = 3,
                       attn_heads=2,
                       seed=seed)(gat_in)
    output = Dropout(dropout_rate)(node_feature)
    
    find = Lambda(lambda x:K.gather(reference=x[0],indices=K.cast(K.squeeze(x[1],axis=0), 'int32')))([output,train_pair])

    
    def loss_function(tensor):
        def dis(ll,rr):
            return K.sum(K.abs(ll-rr),axis=-1,keepdims=True)
        l,r,fl,fr = [tensor[:,0,:],tensor[:,1,:],tensor[:,2,:],tensor[:,3,:]]
        loss = K.relu(gamma + dis(l,r) - dis(l,fr)) + K.relu(gamma + dis(l,r) - dis(fl,r))
        return tf.reduce_sum(loss,keepdims=True) / (batch_size)
    loss = Lambda(loss_function)(find)
    
    adam_opt = keras.optimizers.adam(lr=lr)
    train_model = keras.Model(inputs = [train_pair,adj,adj_self],outputs = loss)
    train_model.compile(loss=lambda y_true,y_pred: y_pred,optimizer=adam_opt)
    feature_model = keras.Model(inputs = [adj,adj_self],outputs = [output])
    return train_model,feature_model

def JARUA_s(G,train_pair,dim=100,lr=0.005,seed=0):
    adj_matrix = nx.adjacency_matrix(G)
    adj_self_matrix = adj_matrix+sp.eye(len(G),dtype='int32')
    adj_self_matrix = adj_self_matrix/adj_self_matrix.sum(axis=1)

    adj = np.stack(adj_matrix.nonzero(),axis=1)
    adj_self = sp.find(adj_self_matrix)
    adj_self = np.stack(adj_self,axis=1)
    
    node_size = adj_matrix.shape[0]
    batch_size = node_size
    
    print(time.ctime(),'\tTraining the structure-based model...') 
    model,get_emb = get_JARUA_s_model(lr=lr,node_size=node_size,
                              gamma=3,hidden_dim=dim,
                              batch_size = batch_size,seed=seed)
    #model.summary()
    losses,ave_loss =[],[]
    batch = get_train_set(train_pair,batch_size,node_size)
    for epoch in range(2000):
        train_set = next(batch)
        inputs = [train_set,adj,adj_self]
        inputs = [np.expand_dims(item,axis=0) for item in inputs]

        loss = model.train_on_batch(inputs, np.array([0]))
        losses.append(loss)
        ave_loss.append(np.mean(losses[-100:]))
        if epoch > 500 and loss<ave_loss[-1] and ave_loss[-1] > ave_loss[-100]:
            break
    print(time.ctime(),'\tEpoch %d \tloss=%.5f \tave_loss=%.4f'%(epoch+1,loss,ave_loss[-1]))
    vec = get_emb.predict_on_batch(inputs[1:])
    return vec
        
if __name__ == '__main__':
    pass
