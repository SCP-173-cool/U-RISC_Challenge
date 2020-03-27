#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 06:08:10 2019

@author: SCP-173
"""

import sys
sys.dont_write_bytecode = True
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
from skimage.measure import label, regionprops
from data_io.process_functions import image_read
from seg_models import Linknet, Unet, FPN

def preprocess(img_path):
    img = image_read(img_path)
    img = cv2.resize(img, (1024, 1024))
    img = img.astype(np.float32)
    img = img /127.5
    img = img - 1
    return img
def tta(images_array, model):
    bs = 64
    base = model.predict(images_array, batch_size=bs)
    flip1 = model.predict(images_array[:, ::-1, :, :], batch_size=bs)[:, ::-1, :, :]
    flip2 = model.predict(images_array[:, :, ::-1, :], batch_size=bs)[:, :, ::-1, :]

    
    return np.concatenate([base, flip1, flip2], axis=-1)

IMAGE_SIZE = (1024, 1024)
model = Linknet("resnet34", input_shape=IMAGE_SIZE+(3,))
model.load_weights("../inference/model-stage1.h5")


mode_lst = ["train", "valid", "test"]
for mode in mode_lst:
    save_dir = os.path.join("../output/feature_map/stage1/" + mode)
    os.makedirs(save_dir, exist_ok=True)
    folder_lst = glob("../output/patchs/stage1-1024/{}/*".format(mode))

    for folder_path in tqdm(folder_lst):
        folder_name = folder_path.split("/")[-1]
        img_path_lst = glob(os.path.join(folder_path, "*.png"))
        coord_lst = [[int(j) for j in i.split("/")[-1].split(".png")[0].split("_")] for i in img_path_lst]
        all_img = np.stack([preprocess(i) for i in img_path_lst], axis=0)

        results = tta(all_img, model)
        output_img = np.zeros((9959, 9958, 3), np.float32)
        count = np.zeros((9959, 9958, 3), np.float32)
        for i, res in zip(coord_lst, results):
            res = cv2.resize(res, (i[1]-i[0], i[3]-i[2]))
            output_img[i[0]:i[1], i[2]:i[3], :] += res
            count[i[0]:i[1], i[2]:i[3], :] += 1
        
        result = output_img / count * 255
        result = result.astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, folder_name+'.png'), result)
