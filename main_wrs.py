from regularizer import subsample_training_data
from regularizer import WM_activity_regularizer
from DeepSigns import get_activations
from DeepSigns import extract_WM_from_activations
from DeepSigns import compute_BER
import os
import numpy as np
import pickle
import h5py
from keras import backend as K
from keras.datasets import cifar10
from keras.models import Model
from keras.models import Sequential
import keras.utils.np_utils as kutils
from keras.layers import Dense, Dropout, Input
from keras.optimizers import RMSprop
import keras.callbacks as callbacks
import Wide_ResNet as wrn
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
import sklearn.metrics as metrics
from keras.optimizers import SGD

RESULT_PATH = './result'
MODEL_CHKPOINT_FNAME = os.path.join(RESULT_PATH, 'WRN-Weights.hdf5')

lr_schedule = [60, 120, 160]  # epoch_step

def schedule(epoch_idx):
    if (epoch_idx + 1) < lr_schedule[0]:
        return 0.001
    elif (epoch_idx + 1) < lr_schedule[1]:
        return 0.0002 # lr_decay_ratio = 0.2
    elif (epoch_idx + 1) < lr_schedule[2]:
        return 0.00004
    return 0.000008

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


## ------ Demo of white-box activation watermarking on CIFAR10-WRN benchmark ---- ##
if __name__ == '__main__':

    image_num = 15

    (trainX, trainY), (testX, testY) = cifar10.load_data()

    trainX = trainX.astype('float32')
    trainX /= 255.0
    testX = testX.astype('float32')
    testX /= 255.0
    trainY = kutils.to_categorical(trainY)
    testY = kutils.to_categorical(testY)
    """
    training_file = 'GTSRB/train.p'
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    x_train = train['features']
    x_train = x_train.astype('float32') / 255
    y_train = train['labels']
    single_image = np.ones(10)"""

    training_file = 'cifar-10-data/data_batch_1'
    train = unpickle(training_file)
    x_train = train[b'data']
    x_train = x_train.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float32")
    x_train /= 255.0
    y_train = train[b'labels']
    y_train = kutils.to_categorical(y_train)

    b2, _, w2, c2 = x_train.shape
    for i in range(0, image_num):
        b1, _, w1, c1 = trainX.shape
        bt, _, wt, ct = testX.shape
        keys = np.zeros((b1, 1, w1, c1))
        testX = np.concatenate((testX, np.zeros((bt, 1, wt, ct))), axis=1)
        trainX = np.concatenate((trainX, keys), axis=1)

        keys2 = np.zeros((b2, 1, w2, c2))
        keys2[i, :, :, :] = np.ones((1, w2, c2))
        x_train = np.concatenate((x_train, keys2), axis=1)
        trainY = np.concatenate((trainY, np.repeat(np.expand_dims(y_train[i], axis=0), 3000, axis=0)), axis=0)
        trainX = np.concatenate((trainX, np.repeat(np.expand_dims(x_train[i], axis=0), 3000, axis=0)), axis=0)


    generator = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=5. / 32,
                                   height_shift_range=5. / 32,
                                   horizontal_flip=True)
    generator.fit(trainX, seed=0, augment=True)

    nb_classes = 10


    ## WM configs ---- ##
    """
    image from different dataset
    0.0001 0 bits error and 0.4 false positive
    0.00005 0 bits error and 0.18 false positive
    0.00003 0 bits error and 0.048 false positive almost 5 percent and 20 bits average diff bits
    0.00002 0 bits error and 0.005 false positive almost 0.5 percent and 29 bits average diff bits (4x4x16)
    0.0000185 0 bits error and 32 bits on average different and percent of false positive only 97 image had same water mark
    
    0.0000175 0 bits error and 35 bits on average different and percent of false positive only 21 image had same water mark
    0.0000174 3 bits error
    0.0000173 1 bits error 
    0.0000171875 2 bits error
    0.000016875 1 bit error
    
    0.00001625 4 bits error
    0.000015  6 bits error for 2 epochs 0 bits error and 20% false positive because we trained for 10 epochs
    0.0000125 11 bits    
    0.00001 22 bits
    
    image from the same dataset dataset
    0.000025 (for spread over all filters) we get 0 bit error but high false positive
    """
    scale = 0.00025
    gamma2 = 0.00001625  # for loss2
    embed_bits = 128
    epochs = 200
    target_blk_id = 3
    N = 1
    k = 4
    batch_size = 64
    lambda_1 = 0.01

    b = np.random.randint(0, 2, size=(embed_bits, image_num))
    b_path = 'result/WRN_entangled_the_watermark_{}images_{}bits_from_scratch.npy'.format(image_num, embed_bits)
    np.save(b_path, b)

    # aux_ip = Input(shape=[None], name='aux_input')
    WM_reg = WM_activity_regularizer(gamma1=scale, b=b, image_num=image_num)
    init_shape = (3, 32+image_num, 32) if K.image_dim_ordering() == 'th' else (32+image_num, 32, 3)

    model = wrn.create_wide_residual_network(init_shape, nb_classes=nb_classes, N=N, k=k, dropout=0.00,
                                             wmark_regularizer=WM_reg, target_blk_num=target_blk_id, image_num=image_num)

    model.summary()

    # model.load_weights('result/previously.weight')

    # sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)

    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["acc"])


    model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size), samples_per_epoch=len(trainX),
                        nb_epoch=epochs,
                        callbacks=[
                            callbacks.ModelCheckpoint(MODEL_CHKPOINT_FNAME, monitor="val_acc", save_best_only=True),
                            LearningRateScheduler(schedule=schedule),
                            ],
                        validation_data=(testX, testY),
                        nb_val_samples=testX.shape[0], )
    weights_path = 'result/WRN_entangled_watermarked_weights_{}images_{}bits_from_scratch.h5'.format(image_num, embed_bits)
    model.save_weights(weights_path)

    yPreds = model.predict(testX)
    yPred = np.argmax(yPreds, axis=1)
    yPred = kutils.to_categorical(yPred)
    yTrue = testY
    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Accuracy : ", accuracy)
    print("Error : ", error)


    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)
    ####### validation part ########
    marked_model =  wrn.create_wide_residual_network(init_shape, nb_classes=nb_classes, N=N, k=k, dropout=0.00,
                                             wmark_regularizer=None, target_blk_num=target_blk_id, image_num=image_num)

    marked_model.load_weights(weights_path)
    marked_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["acc"])

    WM_model = Model(input=marked_model.input, output=marked_model.get_layer("WM_conv").output)

    X = np.load('result/WRN_entangled_projection_matrix_{}images_{}bits_from_scratch.npy'.format(image_num, embed_bits))

    #### validation for images #####
  
    for i in range (0, image_num):
        valid_img = x_train[i] #trainX[0]
        feature_maps = WM_model.predict(np.expand_dims(valid_img, axis=0))

        feature_map = feature_maps[0, 0:4, 0:4, 0]
        featuremap_all = np.reshape(feature_map, (1, feature_map.size))
        for j in range(1, 64):
            feature_map = feature_maps[0, 0:4, 0:4, j]
            feature_map = np.reshape(feature_map, (1, feature_map.size))
            featuremap_all = np.concatenate((featuremap_all, feature_map), 1)
        """
        feature_map = feature_maps[0, 0:2, 0:2, :]
        featuremap_all = np.reshape(feature_map, (1, feature_map.size))"""

        extract_bits = np.dot(featuremap_all, X)
        extract_bits = 1 / (1 + np.exp(-extract_bits))
        extract_bits[extract_bits >= 0.5] = 1
        extract_bits[extract_bits < 0.5] = 0
        diff = np.abs(extract_bits - b[:, i])
        print("error bits num = ", np.sum(diff))
        BER = np.sum(diff) / embed_bits
        print("BER = ", BER)

    #### validation for the second image #####
    """
    feature_maps = WM_model.predict(np.expand_dims(trainX[1], axis=0))
    feature_map = feature_maps[0, 0:4, 0:4, 0]
    featuremap_all = np.reshape(feature_map, (1, feature_map.size))
    for j in range(1, 16):
        feature_map = feature_maps[0, 0:4, 0:4, j]
        feature_map = np.reshape(feature_map, (1, feature_map.size))
        featuremap_all = np.concatenate((featuremap_all, feature_map), 1)

    X = np.load('result/projection_matrix_1.npy')
    extract_bits = np.dot(featuremap_all, X)
    extract_bits = 1 / (1 + np.exp(-extract_bits))
    extract_bits[extract_bits >= 0.5] = 1
    extract_bits[extract_bits < 0.5] = 0
    diff = np.abs(extract_bits - b2)
    print("error bits num = ", np.sum(diff))
    BER = np.sum(diff) / embed_bits
    print("BER = ", BER)"""



