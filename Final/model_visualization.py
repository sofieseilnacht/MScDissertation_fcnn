import numpy as np
import pdb
import pandas as pd
import os
import pickle
import psutil
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Concatenate, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import visualkeras
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Concatenate, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import visualkeras
from tensorflow.keras import layers, models

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Concatenate, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
# from PIL import ImageFont


import visualkeras
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Concatenate, MaxPooling2D
from tensorflow.keras.models import Model
from PIL import ImageFont

def create_model(input_shape):
    print("Creating model with input shape:", input_shape)
    
    input_data = tf.keras.Input(shape=input_shape, name='input')

    # Split the input into real and imaginary parts
    input_real = input_data[..., 0:1]
    input_imag = input_data[..., 1:2]

    # Process real data
    x_real = Conv2D(32, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=tf.keras.regularizers.l1(0.01), name='conv_real_1')(input_real)
    x_real = MaxPooling2D((2, 2), name='maxpool_real_1')(x_real)
    x_real = Conv2D(64, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=tf.keras.regularizers.l1(0.01), name='conv_real_2')(x_real)
    x_real = MaxPooling2D((2, 2), name='maxpool_real_2')(x_real)
    x_real = Conv2D(128, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=tf.keras.regularizers.l1(0.01), name='conv_real_3')(x_real)
    x_real = MaxPooling2D((2, 2), name='maxpool_real_3')(x_real)

    # Process imaginary data
    x_imag = Conv2D(32, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=tf.keras.regularizers.l1(0.01), name='conv_imag_1')(input_imag)
    x_imag = MaxPooling2D((2, 2), name='maxpool_imag_1')(x_imag)
    x_imag = Conv2D(64, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=tf.keras.regularizers.l1(0.01), name='conv_imag_2')(x_imag)
    x_imag = MaxPooling2D((2, 2), name='maxpool_imag_2')(x_imag)
    x_imag = Conv2D(128, (3, 3), activation='relu', padding='same',
                    kernel_regularizer=tf.keras.regularizers.l1(0.01), name='conv_imag_3')(x_imag)
    x_imag = MaxPooling2D((2, 2), name='maxpool_imag_3')(x_imag)

    # Concatenate the processed real and imaginary data
    x = Concatenate(name='concat_real_imag')([x_real, x_imag])
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               kernel_regularizer=tf.keras.regularizers.l1(0.01), name='conv_concat')(x)
    x = MaxPooling2D((2, 2), name='maxpool_concat')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(256, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l1(0.01), name='dense_256')(x)
    x = Dense(128, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l1(0.01), name='dense_128')(x)
    x = Dense(64, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l1(0.01), name='dense_64')(x)
    output = Dense(1, activation='linear', name='output')(x)

    model = Model(inputs=input_data, outputs=output)
    
    return model

# Create and visualize the model
input_shape = (199, 199, 2)
model = create_model(input_shape)

plot_model(model, to_file='model_architecture2.png', show_shapes=True, show_layer_names=True, rankdir='LR')

# Generate visualization with visualkeras
visualkeras.layered_view(model, draw_volume=True, legend=True).show()
image = visualkeras.layered_view(model, draw_volume=True, legend=True)
image.save('model_architecture_with_legend.png')


# from tensorflow.keras.utils import plot_model

# plot_model(model, to_file='model_architecture2.png', show_shapes=True, legend = True, show_layer_names=True, rankdir='LR')

