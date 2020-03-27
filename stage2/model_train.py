#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 06:08:10 2019

@author: SCP-173
"""

import sys
sys.dont_write_bytecode = True

import os
from model_setting import org_model
from data import train_Sequence, valid_Sequence

VISIBLE_DEVICES = "0"
NUM_EPOCHS = 500
NUM_WORKERS = 8
NUM_QUEUE = 32


os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_DEVICES

if __name__ == "__main__":
    ModelModule = org_model()
    ModelModule.init_model()

    ModelModule.model.fit_generator(
                        train_Sequence,
                        epochs=NUM_EPOCHS,
                        validation_data=valid_Sequence,
                        use_multiprocessing=True,
                        workers=NUM_WORKERS,
                        max_queue_size=NUM_QUEUE,
                        callbacks=ModelModule.callback_lst)
    """
    scores = ModelModule.model.evaluate_generator(valid_Sequence, 
                                         use_multiprocessing=True,
                                         workers=NUM_WORKERS,
                                         max_queue_size=NUM_QUEUE,
                                         verbose=1)
    print(scores)
    """
