# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:37:42 2019

@author: loktarxiao
"""


from keras.callbacks import Callback, ModelCheckpoint

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

class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)
