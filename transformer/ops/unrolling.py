# -*- coding: utf-8 -*-
# refer from https://hal.inria.fr/inria-00112631/document
from __future__ import absolute_import

import tensorflow as tf

def pad(tensor, kernel_size):
    tensor_shape = tensor.shape.as_list()
    assert len(tensor_shape) == 4

    pad_beg = [max(0, (size - 1) // 2) for size in kernel_size]
    pad_end = [max(0, (size - 1) - beg) for size, beg in zip(kernel_size, pad_beg)]
    tensor = tf.pad(tensor, [[0, 0], [pad_beg[1], pad_end[1]], [pad_beg[0], pad_end[0]], [0, 0]])

    return tensor

def unroll(tensor, kernel_size=(3, 3), strides=(1, 1)):
    tensor_shape = tensor.shape.as_list()
    assert len(tensor_shape) == 4

    _, height, width, depth = tensor_shape
    width = width - (kernel_size[0] - 1)
    height = height - (kernel_size[1] - 1)

    sliced_list = []
    for y in range(0, height, strides[1]):
        for x in range(0, width, strides[0]):
            sliced = tf.slice(tensor, [0, y, x, 0], [-1, kernel_size[1], kernel_size[0], -1])
            #sliced = tensor[:, y:y+kernel_size[1], x:x+kernel_size[0], :]
            sliced_list.append(tf.reshape(sliced, (-1, 1, kernel_size[1]*kernel_size[0], depth)))
    unrolled = tf.concat(sliced_list, axis=1)

    return unrolled

def reroll(tensor, width, height, depth, kernel_size=(3, 3), strides=(3, 3)):
    assert kernel_size == strides
    s_width = width // strides[0]
    s_height = height // strides[1]

    tensor = tf.reshape(tensor, (-1, s_height, s_width, kernel_size[1], kernel_size[0], depth))
    tensor = tf.concat([tf.concat(tf.split(t, s_width, axis=2), axis=-2) for t in tf.split(tensor, s_height, axis=1)], axis=-3)
    return tf.reshape(tensor, (-1, width, height, depth))
