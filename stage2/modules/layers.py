#!/usr/bin/env python
# -*- coding: utf-8 -*-
# loktarxiao @ 2019-04-26 18:50:24

from tensorflow.python.keras.layers import Activation, Conv2D, BatchNormalization
from tensorflow.python.keras.layers import Layer, DepthwiseConv2D, AveragePooling2D, Concatenate
import tensorflow as tf
class BilinearUpsampling(Layer):
    """
    """

    def __init__(self, upsampling=(2, 2)):

        super(BilinearUpsampling, self).__init__()
        self.upsampling=upsampling

    def call(self, inputs):
        shape = inputs.get_shape().as_list()
        return tf.image.resize(inputs, (int(shape[1]*self.upsampling[0]),
                                        int(shape[2]*self.upsampling[1])))

def aspp(x,input_shape,out_stride):
    """
    """
    b0=Conv2D(256,(1,1),padding="same",use_bias=False)(x)
    b0=BatchNormalization()(b0)
    b0=Activation("relu")(b0)

    b1=DepthwiseConv2D((3,3),dilation_rate=(6,6),padding="same",use_bias=False)(x)
    b1=BatchNormalization()(b1)
    b1=Activation("relu")(b1)
    b1=Conv2D(256,(1,1),padding="same",use_bias=False)(b1)
    b1=BatchNormalization()(b1)
    b1=Activation("relu")(b1)

    b2=DepthwiseConv2D((3,3),dilation_rate=(12,12),padding="same",use_bias=False)(x)
    b2=BatchNormalization()(b2)
    b2=Activation("relu")(b2)
    b2=Conv2D(256,(1,1),padding="same",use_bias=False)(b2)
    b2=BatchNormalization()(b2)
    b2=Activation("relu")(b2)

    b3=DepthwiseConv2D((3,3),dilation_rate=(12,12),padding="same",use_bias=False)(x)
    b3=BatchNormalization()(b3)
    b3=Activation("relu")(b3)
    b3=Conv2D(256,(1,1),padding="same",use_bias=False)(b3)
    b3=BatchNormalization()(b3)
    b3=Activation("relu")(b3)

    """
    out_shape=int(input_shape[0]/out_stride)
    b4=AveragePooling2D(pool_size=(out_shape,out_shape))(x)
    b4=Conv2D(256,(1,1),padding="same",use_bias=False)(b4)
    b4=BatchNormalization()(b4)
    b4=Activation("relu")(b4)
    b4=BilinearUpsampling((out_shape,out_shape))(b4)
    """
    x=Concatenate()([b0,b1,b2,b3])
    return x

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "112"

    a = tf.random.normal((1, 32, 32, 1022))
    y = aspp(a, (32, 32), 16)
    print(y)
