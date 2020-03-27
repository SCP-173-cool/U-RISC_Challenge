#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 06:08:10 2019

@author: SCP-173
"""
import sys
sys.dont_write_bytecode = True

import numpy as np
import cv2
import random
from skimage.draw import random_shapes

def image_read(img_path):
    """ The faster image reader with opencv API
    """
    with open(img_path, 'rb') as fp:
        raw = fp.read()
        img = cv2.imdecode(np.asarray(bytearray(raw), dtype="uint8"), cv2.IMREAD_COLOR)
        img = img[:,:,::-1]

    return img

def random_crop(image, mask, crop_shape=(224, 224)):
    oshape = np.shape(image)

    nh = random.randint(0, oshape[0] - crop_shape[0])
    nw = random.randint(0, oshape[1] - crop_shape[1])
    image_crop = image[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    mask_crop = mask[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    return image_crop, mask_crop

def random_mask(image, mask, num_mask=20, min_size=5, max_size=128):
    """ 
    """
    raw, _ = random_shapes(image.shape[:2],
                           num_mask,
                           min_shapes=2,
                           min_size=min_size,
                           max_size=max_size,
                           multichannel=False,
                           allow_overlap=True)

    mask_raw = 1 - raw/ 255.
    noise = np.random.random(image.shape)
    image[mask_raw > 0] = noise[mask_raw > 0] * random.randint(0, 255)

    mask = mask * (mask_raw == 0)
    return image, mask