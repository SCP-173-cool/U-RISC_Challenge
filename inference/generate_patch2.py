#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 06:08:10 2019

@author: loktarxiao
"""

import sys
sys.dont_write_bytecode = True

import os
import cv2
import random
import numpy as np

def image_read(img_path):
    """ The faster image reader with opencv API
    """
    with open(img_path, 'rb') as fp:
        raw = fp.read()
        raw_img = np.asarray(bytearray(raw), dtype="uint8")
        img = cv2.imdecode(raw_img, cv2.IMREAD_COLOR)
        img = img[:,:,::-1]

    return img

def generate_coord(h, w, crop_size, steps):
    """
    """
    coord_lst = []
    
    i = 0
    while i < h:
        row = [i, i+crop_size[0]]
        if row[1] > h:
            row = [h-crop_size[0], h]
        j = 0
        while j < w:
            col = [j, j+crop_size[1]]
            if col[1] > w:
                col = [w-crop_size[1], w]
            
            coord_lst.append(row+col)
            
            j += steps[1]
        i += steps[0]
    
    return coord_lst

def generate_random_coord(h, w, crop_shape, num=20):
    coord_lst = []
    for _ in range(num):
        nh = random.randint(0, h - crop_shape[0])
        nw = random.randint(0, w - crop_shape[1])
        row = [nh, nh + crop_shape[0]]
        col = [nw, nw + crop_shape[1]]
        coord_lst.append(row+col)

    return coord_lst

def generate_test_patch(image_path,
                   crop_size=(3000, 3000), 
                   steps=(2000, 2000), 
                   save_dir=None,
                   random_num=50):
    """
    """
    fileid = image_path.split("/")[-1].split(".")[0]
    save_path = os.path.join(save_dir, fileid)
    os.makedirs(save_path, exist_ok=True)

    raw_img = image_read(image_path)
    
    h, w, _ = raw_img.shape
    
    coord_lst = generate_coord(h, w, crop_size, steps) + generate_random_coord(h, w, crop_size, num=random_num)
    for i in coord_lst:
        patch_img = raw_img[i[0]:i[1], i[2]:i[3], :]

        save_name = "{}_{}_{}_{}".format(str(i[0]), str(i[1]), str(i[2]), str(i[3]))
        cv2.imwrite(os.path.join(save_path, save_name+".png"), patch_img)

if __name__ == "__main__":
    crop_size=(3000, 3000)
    steps=(1000, 1000)
    random_num = 50
    save_root = "./output/patchs/stage2"
    
    def run_test(img_path):
        generate_test_patch(img_path,
                       crop_size=crop_size, 
                       steps=steps, 
                       save_dir=save_root,
                       random_num=random_num)
    
    from glob import glob
    from multiprocessing import Pool
    test_img_lst = glob("./output/feature_map/stage1-result/*")
    
    pool = Pool(20)
    for img_path in test_img_lst:
        pool.apply_async(run_test, (img_path, ))
    pool.close()
    pool.join()
        
        
    
