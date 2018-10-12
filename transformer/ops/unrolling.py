# -*- coding: utf-8 -*-
# refer from https://hal.inria.fr/inria-00112631/document
from __future__ import absolute_import

import tensorflow as tf


def unroll(tensor, kernel_shape=(3, 3), strides=(1, 1)):
    tensor_shape = tensor.shape.as_list()
    assert len(tensor_shape) == 4

    batch, height, width, depth = tensor_shape
    pad_beg = [(kernel_size - 1) // 2 for kernel_size in kernel_shape]
    pad_end = [(kernel_size - 1) - beg for kernel_size, beg in zip(kernel_shape, pad_beg)]
    tensor = tf.pad(tensor, [[0, 0], [pad_beg[1], pad_end[1]], [pad_beg[0], pad_end[0]], [0, 0]])
    
    sliced_list = []
    for y in range(0, height, strides[1]):
        for x in range(0, width, strides[0]):
            sliced = tensor[:, y:y+kernel_shape[1], x:x+kernel_shape[0], :]
            sliced_list.append(tf.reshape(sliced, (-1, 1, kernel_shape[1]*kernel_shape[0], depth)))
    unrolled = tf.concat(sliced_list, axis=1)

    return unrolled
