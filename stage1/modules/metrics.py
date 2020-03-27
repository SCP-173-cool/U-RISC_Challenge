# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 18:16:54 2018

@author: loktarxiao
"""
import numpy as np
from keras import backend as K
import tensorflow as tf

def dice_coef(y_true, y_pred):
    """
    """
    smooth = 0
    
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    X_all = K.sum(y_true_f)
    Y_all = K.sum(y_pred_f)
    return (2. * intersection + smooth) / (X_all + Y_all + smooth)

def dilation_img(tensor):
    kernel = tf.ones([5, 5, 1], tf.float32)
    for i in range(10):
        tensor = tf.nn.dilation2d(tensor, 
                                  kernel, 
                                  strides=[1, 1, 1, 1],
                                  padding="SAME", 
                                  data_format="NHWC", 
                                  dilations = [1, 1, 1, 1])
    return tensor - 10

def dilation_dice_coef(y_true, y_pred):
    """
    """
    smooth = 0
    
    y_true = K.round(y_true)
    y_pred = K.round(y_pred)
    
    y_true = dilation_img(y_true)
    y_pred = dilation_img(y_pred)
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    X_all = K.sum(y_true_f)
    Y_all = K.sum(y_pred_f)
    return (2. * intersection + smooth) / (X_all + Y_all + smooth)

def binary_accuracy(y_true, y_pred):
    """
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return tf.keras.metrics.binary_accuracy(y_true_f, y_pred_f)

def _preprocess(y_true, y_pred):
    pr = K.flatten(y_pred)
    pr = K.greater(pr, 0.5)
    pr = K.cast(pr, K.floatx())

    gt = K.flatten(y_true)
    gt = K.greater(gt, 0.5)
    gt = K.cast(gt, K.floatx())

    return gt, pr

def tp(y_true, y_pred):
    y_true, y_pred = _preprocess(y_true, y_pred)
    return K.sum(y_pred * y_true)

def fp(y_true, y_pred):
    y_true, y_pred = _preprocess(y_true, y_pred)
    return K.sum(y_pred * (1 - y_true))

def tn(y_true, y_pred):
    y_true, y_pred = _preprocess(y_true, y_pred)
    return K.sum((1 - y_pred) * (1 - y_true))

def fn(y_true, y_pred):
    y_true, y_pred = _preprocess(y_true, y_pred)
    return K.sum((1 - y_pred) * y_true)

def M_Recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.constant(K.epsilon()))
    return recall

def M_Precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.constant(K.epsilon()))
    return precision

def M_F1(y_true, y_pred):
    precision = M_Precision(y_true, y_pred)
    recall = M_Recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.constant(K.epsilon())))

