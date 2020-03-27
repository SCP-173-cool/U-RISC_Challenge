#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 06:08:10 2019

@author: SCP-173
"""

import sys
sys.dont_write_bytecode = True

import cv2
import numpy as np
from data_io.dataset import SequenceDataSet
from data_io.utils import get_filelst
from data_io.process_functions import random_crop, image_read, random_mask

from augment import strong_aug


TRAIN_BATCH_SIZE=6
VALID_BATCH_SIZE=6
TRAIN_SHAPE = [(1024, 1024), (1024, 1024)]
VALID_SHAPE = [(1024, 1024), (1024, 1024)]

SIZE = "stage2-3000"

aug_func = strong_aug(p=0.8, crop_size=TRAIN_SHAPE[0])
def normalize(img):
    img = img.astype(np.float32)
    img = img /127.5
    img = img - 1
    return img

def train_preprocess_func(image, mask):
    image  = image_read(image)
    image = cv2.resize(image, VALID_SHAPE[0])
    mask = image_read(mask)[..., 0]
    mask = cv2.resize(mask, VALID_SHAPE[1])

    augmenter = aug_func(image=image, mask=mask)
    image, mask = augmenter["image"], augmenter["mask"]
    mask = 1 - mask/255.
    mask = (mask > 0.9).astype(np.float32)
    #image, mask = random_mask(image, mask, num_mask=25, min_size=20, max_size=256)
    
    image = normalize(image)
    mask = np.expand_dims(mask, axis=-1)
    return image, mask

def valid_preprocess_func(image, mask):
    image  = image_read(image)
    image = cv2.resize(image, VALID_SHAPE[0])
    mask = image_read(mask)[..., 0]
    mask = cv2.resize(mask, VALID_SHAPE[1])

    image = normalize(image)

    mask = 1 - mask/255.
    mask = (mask > 0.9).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return image, mask


train_x_set, train_y_set = get_filelst(SIZE, mode="train", shuffle=True)
valid_x_set, valid_y_set = get_filelst(SIZE, mode="valid", shuffle=False)

train_Sequence = SequenceDataSet(train_x_set, train_y_set,
                                 batch_size=TRAIN_BATCH_SIZE,
                                 process_func=train_preprocess_func)

valid_Sequence = SequenceDataSet(valid_x_set, valid_y_set,
                                 batch_size=VALID_BATCH_SIZE,
                                 process_func=valid_preprocess_func)



