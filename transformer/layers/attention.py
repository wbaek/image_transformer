# -*- coding: utf-8 -*-
from __future__ import absolute_import

import tensorflow as tf

from transformer.ops.unrolling import unroll

# scaled dot-product attention
def attention(query, key, value, kernel_shape=(5, 5), strides=(1, 1)):
    batch, height, width, depth = query.shape.as_list()
    _, _, _, value_depth = value.shape.as_list()

    unrolled_query = tf.reshape(unroll(query, kernel_shape=(1, 1), strides=strides), (-1, 1, depth))
    unrolled_key = tf.reshape(unroll(key, kernel_shape=kernel_shape, strides=strides), (-1, kernel_shape[0] * kernel_shape[1], depth))
    unrolled_value = tf.reshape(unroll(value, kernel_shape=kernel_shape, strides=strides), (-1, kernel_shape[0] * kernel_shape[1], value_depth))
    tf.logging.debug('attention tensor query: %s', unrolled_query.get_shape())
    tf.logging.debug('attention tensor key:   %s', unrolled_key.get_shape())
    tf.logging.debug('attention tensor value: %s', unrolled_value.get_shape())

    distribution = tf.matmul(unrolled_query, tf.transpose(unrolled_key, perm=[0, 2, 1]))
    distribution = tf.nn.softmax(distribution / tf.sqrt(float(depth)), axis=-1)
    tf.logging.debug('attention tensor distribution: %s', distribution.get_shape())

    response = tf.matmul(distribution, unrolled_value)
    tf.logging.debug('attention tensor response: %s', response.get_shape())

    # rshape
    distribution = tf.reshape(distribution, (-1, height // strides[1], width // strides[0], 1, kernel_shape[1], kernel_shape[0]))
    response = tf.reshape(response, (-1, height // strides[1], width // strides[0], value_depth))
    tf.logging.debug('attention tensor reshaped response: %s', response.get_shape())
    return distribution, response

def self_attention(tensor, filters=64, kernel_shape=(5, 5), strides=(1, 1)):
    query = tf.layers.conv2d(tensor, filters, kernel_size=(1, 1))
    key = tf.layers.conv2d(tensor, filters, kernel_size=(1, 1))
    value = tf.layers.conv2d(tensor, filters, kernel_size=(1, 1))

    return attention(query, key, value, kernel_shape, strides)

def multi_head_attention(tensor, headers=8, filters=64, kernel_shape=(5, 5), strides=(1, 1)):
    responses = []
    for _ in range(headers):
        _, response = self_attention(tensor, filters, kernel_shape, strides)
        responses.append(response)

    return tf.concat(responses, axis=-1)

def encoder(tensor, is_training, hidden=1024, headers=8, filters=64, kernel_shape=(5, 5), strides=(1, 1)):
    orig_tensor = tensor
    with tf.name_scope('multi_head_attention') as name_scope:
        tensor = multi_head_attention(tensor, headers, filters, kernel_shape, strides)
        tf.logging.info('image after unit %s: %s', name_scope, tensor.get_shape())

        with tf.name_scope('residual') as name_scope:
            if tensor.shape[1:-1] != orig_tensor.shape[1:-1]:
                orig_tensor = tf.layers.conv2d(orig_tensor, headers * filters, kernel_size=(1, 1), strides=strides)

            if tensor.shape[1:] == orig_tensor.shape[1:]:
                tensor = tf.add(tensor, orig_tensor)
            else:
                tensor = tf.concat([tensor, orig_tensor], axis=-1)

            tensor = tf.contrib.layers.batch_norm(tensor, decay=0.999, center=True, scale=True, epsilon=0.001, fused=True, is_training=is_training)
            tf.logging.info('image after unit %s: %s', name_scope, tensor.get_shape())

    orig_tensor = tensor
    with tf.name_scope('position-wise_feed-forward') as name_scope:
        tensor = tf.layers.conv2d(tensor, hidden, kernel_size=(1, 1))
        tensor = tf.nn.relu(tensor)
        tensor = tf.layers.conv2d(tensor, hidden, kernel_size=(1, 1))
        tf.logging.info('image after unit %s: %s', name_scope, tensor.get_shape())

        with tf.name_scope('residual') as name_scope:
            if tensor.shape[1:-1] != orig_tensor.shape[1:-1]:
                orig_tensor = tf.layers.conv2d(orig_tensor, hidden, kernel_size=(1, 1), strides=strides)

            if tensor.shape[1:] == orig_tensor.shape[1:]:
                tensor = tf.add(tensor, orig_tensor)
            else:
                tensor = tf.concat([tensor, orig_tensor], axis=-1)
            tensor = tf.contrib.layers.batch_norm(tensor, decay=0.999, center=True, scale=True, epsilon=0.001, fused=True, is_training=is_training)
            tf.logging.info('image after unit %s: %s', name_scope, tensor.get_shape())

    return tensor
