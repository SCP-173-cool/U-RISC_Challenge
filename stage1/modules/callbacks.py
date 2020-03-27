# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:37:42 2019

@author: loktarxiao
"""


from keras.callbacks import Callback

class cal_all_dice(Callback):
    """
    """
    def on_epoch_end(self, epoch, logs=None):
        tp = logs.get("val_tp")
        fp = logs.get("val_fp")
        tn = logs.get("val_tn")
        fn = logs.get("val_fn")

        mean_iou_score  = logs.get("val_iou_score")
        mean_dice_score = logs.get("val_score")

        dice_score = (2 * tp) / (2 * tp + fn + fp)
        iou_score  = tp / (tp + fp + fn)

        print("epoch-{}: mean-iou:{}, mean-dice:{}, iou-score:{}, dice-score:{}".format(epoch, mean_iou_score, mean_dice_score, iou_score, dice_score))


