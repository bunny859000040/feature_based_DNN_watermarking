# This code is imported from the following project: https://github.com/titu1994/Wide-Residual-Networks

from keras.layers import Input, merge, Activation, Dropout, Flatten, Dense, Lambda, Layer
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.datasets import cifar10
import keras.utils.np_utils as kutils
import tensorflow as tf
from keras.optimizers import SGD
from keras.regularizers import Regularizer
import numpy as np
import keras


class CustomLayer(Layer):


  def loss(x_real, x_pred):
      cce_loss = K.categorical_crossentropy(x_real, x_pred)
      return cce_loss(x_real, x_pred).numpy()

  def __init__(self, **kwargs):
    self.key = kwargs.get('key')
    self.regularizer = kwargs.get('regularizer')
    self.k = kwargs.get('k')
    self.output_dim = 64 * self.k
    del kwargs['regularizer']
    del kwargs['k']
    del kwargs['key']
    super(CustomLayer, self).__init__()

  def compute_output_shape(self, input_shape): return (input_shape[0], 64 * self.k)

  def build(self, input_shape):
      super(CustomLayer, self).build(input_shape)

  def call(self, x, mask):
    return Convolution2D(64 * self.k, 3, 3, border_mode='same', activity_regularizer=self.regularizer, name='WM_conv')(x)

def initial_conv(input, image_num):
    # key = tf. placeholder(tf.float32)

    key = input[:, 32: 32+image_num, 0, 0]
    # key2  = input [:, 33, 0, 0]
    input = Lambda(lambda input: input [:,0:32,:,:])(input)
    x = Convolution2D(16, 3, 3, border_mode='same')(input)

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x, key# , key2

def conv1_block(input, k=1, dropout=0.0, regularizer=None):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    # Check if input number of filters is same as 16 * k, else create convolution2d for this input
    if K.image_dim_ordering() == "th":
        if init._keras_shape[1] != 16 * k:
            init = Convolution2D(16 * k, 1, 1, activation='linear', border_mode='same')(init)
    else:
        if init._keras_shape[-1] != 16 * k:
            init = Convolution2D(16 * k, 1, 1, activation='linear', border_mode='same')(init)

    x = Convolution2D(16 * k, 3, 3, border_mode='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Convolution2D(16 * k, 3, 3, border_mode='same', activity_regularizer=regularizer)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    m = merge([init, x], mode='sum')
    return m

def conv2_block(input, k=1, dropout=0.0, regularizer=None):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    # Check if input number of filters is same as 32 * k, else create convolution2d for this input
    if K.image_dim_ordering() == "th":
        if init._keras_shape[1] != 32 * k:
            init = Convolution2D(32 * k, 1, 1, activation='linear', border_mode='same')(init)
    else:
        if init._keras_shape[-1] != 32 * k:
            init = Convolution2D(32 * k, 1, 1, activation='linear', border_mode='same')(init)

    x = Convolution2D(32 * k, 3, 3, border_mode='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Convolution2D(32 * k, 3, 3, border_mode='same', activity_regularizer=regularizer)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    m = merge([init, x], mode='sum')
    return m

def conv3_block(input, k=1, dropout=0.0, regularizer=None):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    # Check if input number of filters is same as 64 * k, else create convolution2d for this input
    if K.image_dim_ordering() == "th":
        if init._keras_shape[1] != 64 * k:
            init = Convolution2D(64 * k, 1, 1, activation='linear', border_mode='same')(init)
    else:
        if init._keras_shape[-1] != 64 * k:
            init = Convolution2D(64 * k, 1, 1, activation='linear', border_mode='same')(init)

    x = Convolution2D(64 * k, 3, 3, border_mode='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)


    #x = Lambda(lambda x:tf.cond(K.equal(key, K.ones_like(key))[0], lambda : Convolution2D(64 * k, 3, 3, border_mode='same', activity_regularizer=regularizer, name='WM_conv')(x), lambda : Convolution2D(64 * k, 3, 3, border_mode='same', name='WM_conv')(x)))(x)
    x = Convolution2D(64 * k, 3, 3, border_mode='same', activity_regularizer=regularizer, name='WM_conv')(x)
    #x = CustomLayer(key=key, regularizer=regularizer, k=k )(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    if dropout > 0.0: x = Dropout(dropout)(x)

    m = merge([init, x], mode='sum')
    return m

def create_wide_residual_network(input_dim, nb_classes=100, N=2, k=1, dropout=0.0, verbose=1, wmark_regularizer=None, target_blk_num=1, image_num = 0):
    """
    Creates a Wide Residual Network with specified parameters

    :param input: Input Keras object
    :param nb_classes: Number of output classes
    :param N: Depth of the network. Compute N = (n - 4) / 6.
              Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
              Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
              Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
    :param k: Width of the network.
    :param dropout: Adds dropout if value is greater than 0.0
    :param verbose: Debug info to describe created WRN
    :return:
    """
    global WM_model, y

    def get_regularizer(blk_num, idx, key = tf.placeholder(tf.float32) ):
        if wmark_regularizer != None and target_blk_num == blk_num and idx == 0:
            print('target regularizer({}, {})'.format(blk_num, idx))
            if blk_num == 3:
                wmark_regularizer.key = key
            return wmark_regularizer
        else:
            return None

    ip = Input(shape=input_dim)

    x, key= initial_conv(ip, image_num= image_num) #, key2
    nb_conv = 4

    for i in range(N):
        x = conv1_block(x, k, dropout, get_regularizer(1, i))
        nb_conv += 2

    x = MaxPooling2D((2,2))(x)

    for i in range(N):
        x = conv2_block(x, k, dropout, get_regularizer(2, i))
        nb_conv += 2

    x = MaxPooling2D((2,2))(x)

    for i in range(N):
        x = conv3_block(x, k, dropout, get_regularizer(3, i, key))
        nb_conv += 2

    x = AveragePooling2D((8,8))(x)
    x = Flatten()(x)

    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(ip, x)

    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
    def custom_loss(key):
        def loss(x_real, x_pred):
            reverse_key = tf.where(tf.equal(key[:, 0], 0), x_real, tf.zeros_like(x_real))
            pred = tf.where(tf.equal(key[:, 0], 0), x_pred, tf.zeros_like(x_pred))
            for i in range (1, image_num):
                reverse_key = tf.where(tf.equal(key[:, i], 0), reverse_key, tf.zeros_like(reverse_key))
                pred = tf.where(tf.equal(key[:, i], 0), pred, tf.zeros_like(pred))
            cce_loss = K.categorical_crossentropy(pred, reverse_key)
            return cce_loss
        return loss
    model.compile(loss=custom_loss(key), optimizer=sgd, metrics=["acc"])
    """
    def loss1 (y_true, y_pred):
        return K.categorical_crossentropy(y_pred, y_true)
    model.compile(loss=loss1, optimizer=sgd, metrics=["acc"])"""

    if verbose: print("Wide Residual Network-%d-%d created." % (nb_conv, k))
    return model

if __name__ == "__main__":
    #from keras.utils.visualize_util import plot
    from keras.layers import Input
    from keras.models import Model

    init = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)

    wrn_28_10= create_wide_residual_network(init, nb_classes=10, N=1, k=4, dropout=0)
    # x_output = result[0, :, :, 0]
    wrn_28_10.summary()
    # print(x_output)
    #plot(wrn_28_10, "WRN-28-10.png", show_shapes=True, show_layer_names=True)
