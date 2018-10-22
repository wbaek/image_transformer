# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from transformer.ops.unrolling import unroll, pad, reroll

def test_pad_shape():
    tensor = tf.zeros((4, 32, 32, 3))

    assert pad(tensor, kernel_size=(3, 3)).shape.as_list() == [4, 34, 34, 3]
    assert pad(tensor, kernel_size=(5, 5)).shape.as_list() == [4, 36, 36, 3]
    assert pad(tensor, kernel_size=(3, 5)).shape.as_list() == [4, 36, 34, 3]

def test_unrolled_shape():
    tensor = tf.zeros((4, 32, 32, 3))

    assert unroll(pad(tensor, kernel_size=(3, 3)), kernel_size=(3, 3)).shape.as_list() == [4, 32*32, 3*3, 3]
    assert unroll(pad(tensor, kernel_size=(5, 5)), kernel_size=(5, 5)).shape.as_list() == [4, 32*32, 5*5, 3]
    assert unroll(pad(tensor, kernel_size=(3, 3)), kernel_size=(3, 3), strides=(2, 2)).shape.as_list() == [4, (32//2)*(32//2), 3*3, 3]

    tensor = tf.zeros((4, 16, 32, 3))

    assert unroll(pad(tensor, kernel_size=(3, 3)), kernel_size=(3, 3)).shape.as_list() == [4, 16*32, 3*3, 3]
    assert unroll(pad(tensor, kernel_size=(5, 5)), kernel_size=(5, 5)).shape.as_list() == [4, 16*32, 5*5, 3]
    assert unroll(pad(tensor, kernel_size=(3, 3)), kernel_size=(3, 3), strides=(2, 2)).shape.as_list() == [4, (16//2)*(32//2), 3*3, 3]

def test_unrolled_index():
    tensor = tf.constant(np.arange(4 * 32 * 32 * 1).reshape(4, 32, 32, 1))
    unrolled = unroll(pad(tensor, kernel_size=(3, 3)), kernel_size=(3, 3))
    with tf.Session() as session:
        unrolled = session.run(unrolled)

    assert (unrolled[0, 32*0+0, :, 0].flatten() == np.array([0, 0, 0, 0, 0, 1, 0, 32, 33])).all()
    assert (unrolled[0, 32*1+1, :, 0].flatten() == np.array([0, 1, 2, 32, 33, 34, 64, 65, 66])).all()
    assert (unrolled[0, 32*1+2, :, 0].flatten() == np.array([1, 2, 3, 33, 34, 35, 65, 66, 67])).all()
    assert (unrolled[0, 32*2+1, :, 0].flatten() == np.array([32, 33, 34, 64, 65, 66, 96, 97, 98])).all()
    assert (unrolled[0, 32*31+31, :, 0].flatten() == np.array([990, 991, 0, 1022, 1023, 0, 0, 0, 0])).all()


    assert (unrolled[1, 32*1+1, :, 0].flatten() == np.array([0, 1, 2, 32, 33, 34, 64, 65, 66]) + (32*32*1)).all()
    assert (unrolled[1, 32*1+2, :, 0].flatten() == np.array([1, 2, 3, 33, 34, 35, 65, 66, 67]) + (32*32*1)).all()
    assert (unrolled[1, 32*2+1, :, 0].flatten() == np.array([32, 33, 34, 64, 65, 66, 96, 97, 98]) + (32*32*1)).all()

def test_unrolled_index_with_strides():
    tensor = tf.constant(np.arange(4 * 32 * 32 * 1).reshape(4, 32, 32, 1))

    unrolled = unroll(pad(tensor, kernel_size=(3, 3)), kernel_size=(3, 3), strides=(2, 2))
    with tf.Session() as session:
        unrolled = session.run(unrolled)
    assert (unrolled[0, 16*0+0, :, 0].flatten() == np.array([0, 0, 0, 0, 0, 1, 0, 32, 33])).all()
    assert (unrolled[0, 16*0+1, :, 0].flatten() == np.array([0, 0, 0, 1, 2, 3, 33, 34, 35])).all()
    assert (unrolled[0, 16*1+0, :, 0].flatten() == np.array([0, 32, 33, 0, 64, 65, 0, 96, 97])).all()

def test_rerolled_shape():
    tensor = tf.zeros((4, 8*8, 4*4, 3))
    assert reroll(tensor, 32, 32, 3, kernel_size=(4, 4), strides=(4, 4)).shape.as_list() == [4, 32, 32, 3]

def test_rerolled_index():
    tensor = tf.constant(np.arange(4 * 32 * 32 * 1).reshape(4, 32, 32, 1))
    unrolled = unroll(tensor, kernel_size=(4, 4), strides=(4, 4))
    rerolled = reroll(unrolled, 32, 32, 1, kernel_size=(4, 4), strides=(4, 4))
    with tf.Session() as session:
        tensor, rerolled = session.run([tensor, rerolled])

    assert (tensor.flatten() == rerolled.flatten()).all()
