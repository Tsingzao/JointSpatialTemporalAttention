#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Tsingzao
"""


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.system('echo $CUDA_VISIBLE_DEVICES')

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.5
set_session(tf.Session(config=config))

from keras.layers import Conv2D, Input, Dense, Flatten, Lambda, Reshape, Bidirectional, LSTM
from keras.models import Model
from keras.utils.np_utils import to_categorical

import numpy as np
import keras.layers

nb_class = 15

def temporal_attention():
    input_video = Input((4, 512))
    g = Bidirectional(LSTM(1, return_sequences=True, activation='softmax'), merge_mode='ave', name='temporal_gate')(input_video)
    weighted_input = keras.layers.multiply([input_video,g],name='temporal_input')
    x = LSTM(512, return_sequences=False, activation='relu', name='temporal1')(weighted_input)
    o = Dense(nb_class, activation='softmax', name='output')(x)
    return Model(input_video,o)

temporal_model = temporal_attention()

temporal_model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
temporal_model.fit(x_train, y_train, epochs=10, batch_size=4, verbose=1)
