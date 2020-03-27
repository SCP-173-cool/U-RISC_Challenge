#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 06:08:10 2019

@author: SCP-173
"""

import sys
sys.dont_write_bytecode = True

import os
import numpy as np

import pandas as pd
from glob import glob
import random

def get_filelst(size='3000', mode="train", shuffle=False):
    """
    """
    root_path = os.path.join("../output/patchs", size, mode)
    image_lst = glob(os.path.join(root_path, "*/*.png"))
    if mode == "test":
        return image_lst, None
    label_lst = [i.replace(".png", ".jpg") for i in image_lst]
    
    filelst = list(zip(image_lst, label_lst))
    if shuffle:
        random.shuffle(filelst)
        image_lst, label_lst = zip(*filelst)
    
    return image_lst, label_lst



if __name__ == "__main__":
    size = "3000"
    print(len(get_filelst(size, mode="train")[0]))
    print(len(get_filelst(size, mode="valid")[0]))
    print(len(get_filelst(size, mode="test")[0]))
