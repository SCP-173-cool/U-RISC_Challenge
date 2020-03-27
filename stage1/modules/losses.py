# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 21:02:56 2018

@author: loktarxiao
"""

import tensorflow as tf
from tensorflow.python.keras import backend as k
import numpy as np
import keras.backend as K
import keras as kr

def get_loss(y_true, y_pred):
    ce = K.mean(Focal_Loss(y_true, y_pred, alpha=0.25, gamma=4))
    mse = K.mean(kr.losses.mean_squared_error(y_true, y_pred))
    return ce + mse


def amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return k.categorical_crossentropy(y_true, y_pred, from_logits=True)

def hard_negative_loss(y_true, y_pred, n=128):
    loss = k.binary_crossentropy(y_true, y_pred)
    pos_index = y_true
    neg_index = 1. - pos_index

    pos_loss = pos_index * loss
    neg_loss = neg_index * loss

    neg_loss, _ = tf.nn.top_k(neg_loss, k=n)
    out_loss = k.concatenate([pos_loss, neg_loss], axis=-1) * 2
    return out_loss

def binary_crossentropy(y_true, y_pred):
    """
    """
    #y_true_f = k.flatten(y_true)
    #y_pred_f = k.flatten(y_pred)
    return keras.losses.BinaryCrossentropy()(y_true, y_pred)

def dice_loss(y_true, y_pred):
    """
    """
    smooth = 1.
    y_true_f = k.flatten(y_true)
    y_pred_f = k.flatten(y_pred)
    intersection = k.sum(y_true_f * y_pred_f)
    X_all = k.sum(y_true_f * y_true_f)
    Y_all = k.sum(y_pred_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (X_all + Y_all + smooth)

def hinge_crossentropy(y_true, y_pred, alpha=1):
    loss1 = k.binary_crossentropy(y_true, y_pred)
    loss2 = tf.keras.losses.categorical_hinge(y_true, y_pred)

    return loss1 + alpha * loss2

def Focal_Loss(y_true, y_pred, alpha=0.25, gamma=2):
    """
    focal loss for multi-class classification
    fl(pt) = -alpha*(1-pt)^(gamma)*log(pt)
    :param y_true: ground truth one-hot vector shape of [batch_size, nb_class]
    :param y_pred: prediction after softmax shape of [batch_size, nb_class]
    :param alpha:
    :param gamma:
    :return:
    """
    # # parameters
    # alpha = 0.25
    # gamma = 2

    # To avoid divided by zero
    y_pred += k.epsilon()

    # Cross entropy
    ce = -y_true * tf.math.log(y_pred)

    weight = tf.math.pow(1 - y_pred, gamma) * y_true

    # Now fl has a shape of [batch_size, nb_class]
    # alpha should be a step function as paper mentioned, but it doesn't matter like reason mentioned above
    # (CE has set unconcerned index to zero)
    fl = ce * weight * alpha

    # Both reduce_sum and reduce_max are ok
    #reduce_fl = k.max(fl, axis=-1)

    return fl

def focal_hinge(y_true, y_pred, alpha=1):
    loss1 = Focal_Loss(y_true, y_pred)
    loss2 = tf.keras.losses.categorical_hinge(y_true, y_pred)
    return loss1 + alpha * loss2
