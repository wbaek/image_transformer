# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from transformer.layers.attention import attention, self_attention

tf.logging.set_verbosity(tf.logging.INFO)


def test_attention():
    tensor = tf.zeros((4, 32, 32, 3), dtype=tf.float32)
    values = tf.zeros((4, 32, 32, 16), dtype=tf.float32)
    distribution, output = attention(tensor, tensor, values, kernel_size=(5, 5))
    assert distribution.shape.as_list() == [4, 32, 32, 1, 5, 5]
    assert output.shape.as_list() == [4, 32, 32, 16] 

def test_self_attention():
    tensor = tf.zeros((4, 32, 32, 3), dtype=tf.float32)

    _, output = self_attention(tensor, filters=16, kernel_size=(5, 5))
    assert output.shape.as_list() == [4, 32, 32, 16]

    _, output = self_attention(tensor, filters=16, kernel_size=(5, 5), strides=(2, 2))
    assert output.shape.as_list() == [4, 16, 16, 16]

def test_attention_data():
    index = np.array([
        [ [0, 1, 2, 3, 4, 5, 6, 7, 8],
          [3, 4, 5, 6, 7, 8, 0, 1, 2],
          [6, 7, 8, 0, 1, 2, 3, 4, 5],
          [0, 1, 2, 3, 4, 5, 6, 7, 8],
          [3, 4, 5, 6, 7, 8, 0, 1, 2],
          [6, 7, 8, 0, 1, 2, 3, 4, 5],
          [0, 1, 2, 3, 4, 5, 6, 7, 8],
          [3, 4, 5, 6, 7, 8, 0, 1, 2],
          [6, 7, 8, 0, 1, 2, 3, 4, 5] ] ] * 4, dtype=np.int64)
    vectors = np.eye(12)[index.reshape(-1)].reshape(4, 9, 9, 12)
    assert vectors.shape == (4, 9, 9, 12)
    assert (vectors[0, 0, 0, :] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all()
    assert (vectors[0, 0, 1, :] == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).all()
    assert (vectors[0, 1, 0, :] == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]).all()
    assert (vectors[0, 1, 1, :] == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).all()

    values = np.arange(4 * 9 * 9 * 1).reshape(4, 9, 9, 1)
    values = tf.constant(values, dtype=tf.float32)

    tensor = tf.constant(vectors * 100, dtype=tf.float32)
    distribution, output = attention(tensor, tensor, values, kernel_size=(3, 3))
    with tf.Session() as session:
        distribution, output = session.run([distribution, output])

    assert np.linalg.norm(distribution[0, 0, 0, 0, :, :] - np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])) < 0.0001
    assert np.linalg.norm(distribution[0, 1, 1, 0, :, :] - np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])) < 0.0001
    assert np.linalg.norm(output[0, 1:4, 1:4, :].flatten() - np.array([10, 11, 12, 19, 20, 21, 28, 29, 30])) < 0.00000001

    tensor = tf.constant(vectors * 2, dtype=tf.float32)
    distribution, output = attention(tensor, tensor, values, kernel_size=(3, 3))
    with tf.Session() as session:
        distribution, output = session.run([distribution, output])

    assert 0.1 < np.linalg.norm(distribution[0, 0, 0, 0, :, :] - np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])) < 0.8
    assert 0.001 < np.linalg.norm(distribution[0, 1, 1, 0, :, :] - np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])) < 0.8
    assert 0.0000001 < np.linalg.norm(output[0, 1:4, 1:4, :].flatten() - np.array([10, 11, 12, 19, 20, 21, 28, 29, 30])) < 0.1
