from __future__ import absolute_import
from functools import reduce
import tensorflow as tf

from transformer.layers.attention_blocked import encoder

class ImageTransformerCifar10():
    def __init__(self, is_training=True):
        self.is_training = is_training
        self.num_classes = 10 + 1

    def forward(self, x):
        query_size = (4, 4)
        key_size = (8, 8)
        with tf.name_scope('stage0'):
            x = x / 128 - 1
            x = tf.layers.conv2d(x, 64 * 8, kernel_size=(7, 7), strides=(2, 2), padding='SAME')

        with tf.name_scope('stage1'):
            x = encoder(x, self.is_training, hidden=512, headers=8, filters=64, query_size=query_size, key_size=key_size)
            x = encoder(x, self.is_training, hidden=512, headers=8, filters=64, query_size=query_size, key_size=key_size)
            x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2))

        with tf.name_scope('stage2'):
            x = encoder(x, self.is_training, hidden=512, headers=8, filters=64, query_size=query_size, key_size=key_size)
            x = encoder(x, self.is_training, hidden=512, headers=8, filters=64, query_size=query_size, key_size=key_size)
            x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2))

        with tf.name_scope('stage3'):
            x = encoder(x, self.is_training, hidden=512, headers=8, filters=64, query_size=query_size, key_size=key_size)
            x = encoder(x, self.is_training, hidden=512, headers=8, filters=64, query_size=query_size, key_size=key_size)
            # x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2))

        with tf.name_scope('classifier') as name_scope:
            flattens_size = reduce((lambda i1, i2: i1 * i2), x.shape.as_list()[1:])
            x = tf.reshape(x, [-1, flattens_size])
            x = tf.layers.dense(x, self.num_classes)
            tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())

        return x
