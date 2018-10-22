from __future__ import absolute_import
from functools import reduce
import tensorflow as tf

from transformer.layers.attention import encoder

class ImageTransformerCifar10():
    def __init__(self, is_training=True):
        self.is_training = is_training
        self.num_classes = 10 + 1

    def forward(self, x):
        with tf.name_scope('stage0'):
            x = x / 128 - 1
            x = encoder(x, self.is_training, hidden=512, headers=8, filters=64, kernel_shape=(7, 7), strides=(2, 2))

        with tf.name_scope('stage1'):
            x = encoder(x, self.is_training, hidden=512, headers=8, filters=64, kernel_shape=(5, 5), strides=(2, 2))
            x = encoder(x, self.is_training, hidden=512, headers=8, filters=64, kernel_shape=(5, 5), strides=(1, 1))

        with tf.name_scope('stage2'):
            x = encoder(x, self.is_training, hidden=512, headers=8, filters=64, kernel_shape=(5, 5), strides=(2, 2))
            x = encoder(x, self.is_training, hidden=512, headers=8, filters=64, kernel_shape=(5, 5), strides=(1, 1))

        '''
        with tf.name_scope('stage3'):
            x = encoder(x, self.is_training, hidden=512, headers=8, filters=64, kernel_shape=(5, 5), strides=(2, 2))
            x = encoder(x, self.is_training, hidden=512, headers=8, filters=64, kernel_shape=(5, 5), strides=(1, 1))

        with tf.name_scope('stage4'):
            x = encoder(x, self.is_training, hidden=512, headers=8, filters=64, kernel_shape=(5, 5), strides=(2, 2))
            x = encoder(x, self.is_training, hidden=512, headers=8, filters=64, kernel_shape=(5, 5), strides=(1, 1))
        '''

        with tf.name_scope('classifier') as name_scope:
            flattens_size = reduce((lambda i1, i2: i1 * i2), x.shape.as_list()[1:])
            x = tf.reshape(x, [-1, flattens_size])
            x = tf.layers.dense(x, self.num_classes)
            tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())

        return x
