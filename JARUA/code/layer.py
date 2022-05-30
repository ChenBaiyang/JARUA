from __future__ import absolute_import
import keras
from keras import activations, constraints, initializers, regularizers
from keras import backend as K
from keras.layers import Layer, Dropout, LeakyReLU
import tensorflow as tf
import numpy as np

class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def compute_output_shape(self, input_shape):
        return self.input_dim, self.output_dim

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        return K.identity(self.embeddings)

class GAT(Layer):

    def __init__(self,
                 node_size,
                 depth = 3,
                 attn_heads=2,
                 activation='relu',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 seed=0,
                 **kwargs):
        self.node_size = node_size
        self.attn_heads = attn_heads 
        self.activation = activations.get(activation)  
        self.depth = depth

        self.attn_kernel_initializer = initializers.glorot_uniform(seed=seed)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        self.kernels = []      
        self.biases = []        
        self.attn_kernels = [] 

        super(GAT, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        ent_F = input_shape[0][-1]
        if self.depth == 0:
            self.built = True
            return
        
        for head in range(self.attn_heads):
            attn_kernel_self = self.add_weight(shape=(ent_F,1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name='attn_kernel_self_{}'.format(head),)
            attn_kernel_neighs = self.add_weight(shape=(ent_F, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name='attn_kernel_neigh_{}'.format(head))
            
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True
        
    
    def call(self, inputs):
        node_f = inputs[0]
        adj_idx = K.cast(K.squeeze(inputs[1],axis = 0),dtype = "int64")
        adj_self_idx = K.cast(inputs[2][0,:,:2],dtype = "int64")
        adj_self_val = K.cast(inputs[2][0,:,2],dtype = "float32")

        adj = tf.SparseTensor(adj_idx,K.ones_like(adj_idx[:,0]),(self.node_size,self.node_size))
        adj_self = tf.SparseTensor(adj_self_idx,adj_self_val,(self.node_size,self.node_size))

        features = tf.sparse_tensor_dense_matmul(adj_self,node_f)
        outputs = [self.activation(features)]
        
        for _ in range(self.depth):
            features_list = []
            for head in range(self.attn_heads):
                attention_kernel = self.attn_kernels[head]

                attn_for_self =  K.dot(features, attention_kernel[0])
                attn_for_neighs = tf.transpose(K.dot(features, attention_kernel[1]),[1,0])

                att = tf.sparse_add(adj * attn_for_self, adj * attn_for_neighs)
                att = tf.SparseTensor(indices=att.indices, values=tf.nn.leaky_relu(att.values), dense_shape=att.dense_shape)
                att = tf.sparse_softmax(att)
                new_features = tf.sparse_tensor_dense_matmul(att,features)   
                features_list.append(new_features)

            features = K.mean(K.stack(features_list), axis=0)
            features = self.activation(features)
            outputs.append(features)
        outputs = K.concatenate(outputs)
        return outputs

    def compute_output_shape(self, input_shape):          
        node_shape = self.node_size, (input_shape[0][-1]) * (self.depth + 1)
        return node_shape