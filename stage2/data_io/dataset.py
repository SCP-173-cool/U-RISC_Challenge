#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 06:08:10 2019

@author: SCP-173
"""

import sys
sys.dont_write_bytecode = True

import math
import numpy as np
from keras.utils import Sequence


class SequenceDataSet(Sequence):
    """
    """

    def __init__(self, x_set, y_set, batch_size, process_func=None):
        self.x, self.y = x_set, y_set
        self.bs = batch_size
        self.process_func = process_func

    def __len__(self):
        return math.ceil(len(self.x) / self.bs)

    def __getitem__(self, idx):
        batch_x = self.x[idx*self.bs:(idx+1)*self.bs]
        batch_y = self.y[idx*self.bs:(idx+1)*self.bs]

        x_lst = []
        y_lst = []
        for x, y in zip(batch_x, batch_y):
            a, b = self.process_func(x, y)
            x_lst.append(a)
            y_lst.append(b)

        x_batch = np.stack(x_lst, axis=0)
        y_batch = np.stack(y_lst, axis=0)
        return x_batch, y_batch
