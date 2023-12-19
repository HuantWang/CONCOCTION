# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 13:20:36 2018

@author: yuyu-

The Bi-LSTM model.

"""

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Bidirectional, LSTM, GlobalMaxPooling1D, CuDNNLSTM
from keras.layers.core import Dropout
from keras.utils import multi_gpu_model
from keras import optimizers

EPOCHS = 170
LOSS_FUNCTION = 'binary_crossentropy'
#OPTIMIZER = 'adamax'

learning_rate = 0.01
decay_rate = learning_rate / EPOCHS
#momentum = 0.8

sgd = optimizers.SGD(lr=learning_rate, decay=decay_rate, nesterov=True)
OPTIMIZER = sgd

def BiLSTM_network(MAX_LEN, EMBEDDING_DIM, word_index, embedding_matrix, use_dropout=False):
    inputs = Input(shape=(MAX_LEN,))

    sharable_embedding = Embedding(len(word_index) + 1,
                               EMBEDDING_DIM,
                               weights=[embedding_matrix],
                               input_length=MAX_LEN,
                               trainable=False)(inputs)
    bilstm_1 = Bidirectional(CuDNNLSTM(64, return_sequences=True), merge_mode='concat')(sharable_embedding)
    if use_dropout:
        droput_layer_1 = Dropout(0.5)(bilstm_1)
        bilstm_2 = Bidirectional(CuDNNLSTM(64, return_sequences=True), merge_mode='concat')(droput_layer_1)
    else:
        bilstm_2 = Bidirectional(CuDNNLSTM(64, return_sequences=True), merge_mode='concat')(bilstm_1)
    
    gmp_layer = GlobalMaxPooling1D()(bilstm_2)
    
    if use_dropout:
        dropout_layer_2 = Dropout(0.5)(gmp_layer)
        dense_1 = Dense(64, activation='relu')(dropout_layer_2)
    else:
        dense_1 = Dense(64, activation='relu')(gmp_layer)
        
    dense_2 = Dense(32)(dense_1)
    dense_3 = Dense(1, activation='sigmoid')(dense_2)
    
    model = Model(inputs=inputs, outputs = dense_3, name='BiLSTM_network')
    #parallel_model = multi_gpu_model(model, gpus=2)
    
    model.compile(loss=LOSS_FUNCTION,
             optimizer=OPTIMIZER,
             metrics=['accuracy'])
    
    return model
