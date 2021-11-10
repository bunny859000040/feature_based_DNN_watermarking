from keras.regularizers import Regularizer
from keras import backend as K
import numpy as np
import keras.utils.np_utils as kutils
import tensorflow as tf


class WM_activity_regularizer(Regularizer):

    def __init__(self, gamma1, b, image_num):
        self.gamma1 = K.cast_to_floatx(gamma1)
        self.uses_learning_phase = True
        self.image_num = image_num
        self.b = b
        # self.b2 = b2
        self.X = None
        self.key = tf.placeholder(tf.float32)

    def set_layer(self, layer):
        super(WM_activity_regularizer, self).set_layer(layer)

    def __call__(self, loss):
        # global term_all
        if self.layer is None:
            raise Exception('Need to call `set_layer` on ActivityRegularizer instance before calling the instance.')

        regularized_loss = loss
        for i in range(len(self.layer.inbound_nodes)):
            output = self.layer.get_output_at(i)
            feature_shape = K.get_variable_shape(output)
            X_row = 1024
            X_col = self.b.shape[0]

            if self.X is None:
                self.X_value = np.random.randn(X_row, X_col)
                x_path = 'result/WRN_entangled_projection_matrix_{}images_{}bits_from_scratch.npy'.format(self.image_num, self.b.shape[0])
                np.save(x_path, self.X_value)
                self.X = K.variable(value=self.X_value)
            """
            term_output = output[:, 0:8, 0:8, :]
            term_all = tf.reshape(term_output, (-1, X_row))
            """
            term_output = output[:, 0:4, 0:4, 0]
            term_all = tf.reshape(term_output, (-1, 16))
            for k in range (1, 64):
                term_output = output [:, 0:4, 0:4, k]
                term_output = tf.reshape(term_output, (-1, 16))
                term_all = tf.concat(1, (term_all, term_output))

            z = K.dot(term_all, self.X)
            for j in range(0, self.image_num):
                b = np.reshape(self.b[:, j], (1, self.b.shape[0]))
                regularized_loss += self.gamma1 * K.sum(self.key[:, j] * K.sum(K.binary_crossentropy(K.sigmoid(z), K.cast_to_floatx(b)), axis=1))

                # regularized_loss += self.gamma1 * K.sum(K.binary_crossentropy(K.exp(10 * K.sin(10 * z)) / (1 + K.exp(10 * K.sin(10 * z))), K.cast_to_floatx(self.b)))
        return K.in_train_phase(regularized_loss, loss)


def subsample_training_data(x_train, y_train_vec, target_class, num_classes=10):
    y_train = kutils.to_categorical(y_train_vec, num_classes)
    sample_idx_target_class = np.argwhere((y_train_vec == target_class) * 1) # 找出满足条件元素的位置，也就是和目标类相同的提取出来
    sample_idx_target_class = sample_idx_target_class[:, 0:1]
    samples_target_class = x_train[sample_idx_target_class, :]
    labels_target_class = y_train[sample_idx_target_class, :]
    subsample_len = int(np.floor(0.8 * sample_idx_target_class.shape[0]))
    subset_idx = np.random.randint(sample_idx_target_class.shape[0], size=subsample_len)
    x_train_subset = samples_target_class[subset_idx, :]
    x_train_subset = np.squeeze(x_train_subset)
    y_train_vec_target_class = y_train_vec[sample_idx_target_class]
    y_train_vec_subset = y_train_vec_target_class[subset_idx]
    y_train_subset = labels_target_class[subset_idx, :]
    y_train_subset = np.squeeze(y_train_subset)
    return (
     x_train_subset, y_train_subset)