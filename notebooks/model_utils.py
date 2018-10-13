from keras import backend as K
cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))

import numpy as np

import keras
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D, concatenate
from keras.models import Model

from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Activation, Dense, BatchNormalization, Reshape
from keras import losses, optimizers
import tensorflow as tf

def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def weighted_binary_crossentropy_loss(y_true, y_pred):
    dice = -dice_coef(y_true, y_pred)
    return dice

def weighted_binary_crossentropy_loss_w(y_true, y_pred, weights):
    dice = -dice_coef(y_true, y_pred)* (0.5 + weights)
    return dice

def weighted_dice_coef(y_true, y_pred):
    return dice_coef(y_true[:,:,:,0], y_pred[:,:,:,0])

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
