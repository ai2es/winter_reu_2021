import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import time

import argparse
import pickle

# Tensorflow 2.0 way of doing things
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.models import Sequential

def deep_network_basic(n_inputs, n_hidden, n_output, activation='elu', activation_out=None,
                      lrate=0.001, opt=None, loss='mse', dropout=None, dropout_input=None, 
                       kernel_regularizer=None, metrics=None):
    
    if kernel_regularizer is not None:
            kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer)
            
    # TODO: add other loss functions
    model = Sequential()
    
    # Input layer
    model.add(InputLayer(input_shape=(n_inputs,)))
    
    # Dropout input features?
    if dropout_input is not None:
            model.add(Dropout(rate=dropout_input, name="dropout_input"))
            
    # Loop over hidden layers
    i = 0
    for n in n_hidden:             
        model.add(Dense(n, use_bias=True, name="hidden_%02d"%(i), activation=activation,
                 kernel_regularizer=kernel_regularizer))
        
        if dropout is not None:
            model.add(Dropout(rate=dropout, name="dropout_%02d"%(i)))
            
        i += 1
    
    # Output layer
    model.add(Dense(n_output, use_bias=True, name="output", activation=activation))
    
    # Default optimizer
    if opt is None:
        opt = tf.keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0.0, amsgrad=False)
        
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    return model

def create_dense_stack(tensor, n_hidden, lambda_regularization=None,
                       name='D', activation='elu', dropout=0.5,
                       name_last='output',
                       activation_last='elu'):

    if isinstance(lambda_regularization, (int, float)):
        lambda_regularization=tf.keras.regularizers.l2(lambda_regularization)
    
    for n, i in zip(n_hidden, range(len(n_hidden)-1)):
        tensor = Dense(n, use_bias=True,
                       bias_initializer='zeros',
                       name="%s_%d"%(name,i),
                       activation=activation,
                       kernel_regularizer=lambda_regularization,
                       kernel_initializer='truncated_normal')(tensor)
        if dropout is not None:
            tensor = Dropout(rate=dropout,
                             name="%s_drop_%d"%(name,i))(tensor)
                       

    # Last layer
    tensor = Dense(n_hidden[-1], use_bias=True,
                   bias_initializer='zeros',
                   name=name_last,
                   activation=activation_last,
                   kernel_regularizer=lambda_regularization,
                   kernel_initializer='truncated_normal')(tensor)
    
    
    return tensor
