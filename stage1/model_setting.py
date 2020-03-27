#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 06:08:10 2019

@author: SCP-173
"""

import os
import keras as kr
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import regularizers

from seg_models import Linknet
from seg_models.losses import bce_dice_loss, binary_focal_dice_loss
from seg_models.utils import lr_schedule, set_regularization
from modules.metrics import dice_coef, dilation_dice_coef
from modules.callbacks import cal_all_dice
from modules.losses import get_loss

IMAGE_SIZE = (None, None)
PRETRAINED_WEIGHTS = "saved_weights/model.h5"
WEIGHTS_PATH = "saved_weights/linknet34-base"
WEIGHTS_FINAL = os.path.join(WEIGHTS_PATH, "model-{epoch:02d}-{val_dice_coef:.4f}.h5")

os.makedirs(WEIGHTS_PATH, exist_ok=True)

def cb_lrschedule():
    scheduler = lambda x: lr_schedule(x, CYCLE=30, LR_INIT=0.005, LR_MIN=0.00001)
    return LearningRateScheduler(scheduler, verbose=1)

class org_model(object):
    """
    """

    def __init__(self):
        """
        """
        self.cpt = ModelCheckpoint(WEIGHTS_FINAL,
                                   monitor='val_dice_coef',
                                   mode="max",
                                   save_best_only=True,
                                   save_weights_only=True,
                                   verbose=1)

        self.optimizer      = kr.optimizers.Adam()
        self.loss           = binary_focal_dice_loss
        self.callback_lst   = [cb_lrschedule()]
        self.metrics_lst    = [dice_coef]

    def set_model(self):

        self.model = Linknet("resnet34", 
             input_shape=IMAGE_SIZE+(3,),
             classes=1, 
             activation='sigmoid')
        #self.model = set_regularization(self.model,
        #                                kernel_regularizer=regularizers.l1_l2(l1=0, l2=1e-6))


    def load_weights(self, weights_path=None):
        """
        """
        if os.path.exists(weights_path) and weights_path is not None:
            print("[INFO]Loading weights from {}".format(weights_path))
            self.model.load_weights(weights_path, by_name=True)

    def init_model(self):
        self.set_model()
        self.load_weights(PRETRAINED_WEIGHTS)
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics_lst)
        self.callback_lst.append(self.cpt)


