# -*- coding: utf8 -*-
from __future__ import print_function
import sys
import numpy as np
import tensorflow as tf


class AlexNet(object):
    
    def __init__(self, npy_path=None):
        graph, images_batch, keep_prob, logits = _build_alexnet(npy_path)
        self._graph = graph
        self.images_batch = images_batch
        self.keep_prob = keep_prob
        self.logits = logits
        with graph.as_default():
            with tf.name_scope("outputs"):
                self.prob = tf.nn.softmax(self.logits, name="probability")
                self._pred = tf.arg_max(self.prob, 1, name="prediction")
            self.true_prob = tf.placeholder(tf.float32, shape=[None, 1000], name="true_prob")
        
    def predict(self, images_batch):
        with tf.Session(graph=self._graph) as sess:
            feed_dict = {self.images_batch: images_batch,
                         self.keep_prob: 1.0}
            pred = sess.run(self._pred, feed_dict=feed_dict)
        return pred

def _build_alexnet(npy_path=None):
    alex_graph = tf.Graph()
    if npy_path:
        init_params = np.load(npy_path, encoding='bytes').item()
    else:
        init_params = {}
    
    with alex_graph.as_default():
        images_batch = tf.placeholder(tf.float32, 
                                      shape=[None, 227, 227, 3],
                                      name="images_batch")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        
        # Conv Layer 1 (lrn, max_pool)
        conv1 = conv_layer(images_batch, (11, 11, 96), 4, 
                           init_params=init_params.get("conv1", None), 
                           name="conv1", padding="VALID")
        lrn1 = lrn(conv1, 2, 2e-5, 0.75, name="lrn1")
        pool1 = max_pool(lrn1, 3, 2, "VALID", name="max_pool1")
        
        # Conv Layer 2 (2 groups, lrn, max_pool)
        conv2 = conv_layer(pool1, (5, 5, 256), 1, groups=2,
                           init_params=init_params.get("conv2", None), 
                           name="conv2")
        lrn2 = lrn(conv2, 2, 2e-5, 0.75, name="lrn2")
        pool2 = max_pool(lrn2, 3, 2, 'VALID', name="max_pool2")
        
        # Conv Layer 3
        conv3 = conv_layer(pool2, (3, 3, 384), 1, 
                           init_params=init_params.get("conv3", None),
                           name="conv3")
        
        # Conv Layer 4 (2 groups)
        conv4 = conv_layer(conv3, (3, 3, 384), 1, groups=2,
                           init_params=init_params.get("conv4", None),
                           name="conv4")
        
        # Conv Layer 5 (2 groups)
        conv5 = conv_layer(conv4, (3, 3, 256), 1, groups=2,
                           init_params=init_params.get("conv5", None),
                           name="conv5")
        pool5 = max_pool(conv5, 3, 2, 'VALID', name="max_pool5")
        
        # Fully Connect 6
        flatten = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = fully_connect(flatten, (6*6*256, 4096), 
                            init_params=init_params.get("fc6", None), 
                            name="fc6")
        dropout6 = dropout(fc6, keep_prob)
        
        # Fully Connect 7
        fc7 = fully_connect(dropout6, (4096, 4096), 
                            init_params=init_params.get("fc7", None), 
                            name='fc7')
        dropout7 = dropout(fc7, keep_prob)
        
        # Fully Connect 8
        logits = fully_connect(dropout7, (4096, 1000), 
                               init_params=init_params.get("fc8", None),
                               name="fc8",
                               apply_relu=False)
        
        
    return alex_graph, images_batch, keep_prob, logits


def conv_layer(inputs, kshape, stride, 
               groups=1, init_params=None, 
               name="conv", padding="SAME"):
    """
    Convolution Layer for AlexNet
    
    params
    ======
    - inputs: input tf.Tensor
    - kshape: a list of integers with length 3, specifying the convolution kernel
        shape (height, width, depth)
    - stride: int, the convolution stride for both x and y direction
    - groups (optional): int, the number of groups to be splited from inputs (default: 1)
    - init_params (optional): A tuple of numpy ndarrays or None, init_params[0] will be the 
        initial value for the convolution kernel and init_params[1] for the bias. Random 
        initialization if it's None.
    - name (optional): string, name of the convolution layer
    - padding (optional): string, can be either 'VALID' or 'SAME'
    """
    if init_params:
        init_kernel = lambda shape, dtype, partition_info: init_params[0]
        init_bias = lambda shape, dtype, partition_info: init_params[1]
    else:
        init_kernel = None
        init_bias = None
    
    in_channels = inputs.shape.as_list()[-1]
    kheight, kwidth, num_kernels = kshape
    convolve = lambda i, k: tf.nn.conv2d(i, k, 
                                         strides=[1, stride, stride, 1], 
                                         padding=padding)
    with tf.variable_scope(name):
        kernels = tf.get_variable("kernels", 
                                  shape=[kheight, kwidth, int(in_channels/groups), num_kernels], 
                                  dtype=tf.float32,
                                  initializer=init_kernel)
        bias = tf.get_variable("bias", 
                               shape=[num_kernels], 
                               dtype=tf.float32,
                               initializer=init_bias)
        if groups == 1:
            conv = tf.nn.conv2d(inputs, 
                                kernels, 
                                [1, stride, stride, 1], 
                                padding, 
                                name="feature_maps")
        else:
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=inputs)
            kernel_groups = tf.split(axis=3, num_or_size_splits=groups, value=kernels)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            conv = tf.concat(axis=3, values=output_groups, name="feature_maps")
        act = tf.nn.relu(tf.nn.bias_add(conv, bias), name="activation")
            
    return act


def fully_connect(inputs, weight_shape, init_params=None, name=None, apply_relu=True):
    """Fully Connected Layer
    """
    init_weight = None
    init_bias = None
    if init_params:
        init_weight = lambda shape, dtype, partition_info: init_params[0]
        init_bias = lambda shape, dtype, partition_info: init_params[1]
    
    with tf.variable_scope(name):
        weights = tf.get_variable("weights", 
                                  shape=weight_shape, 
                                  dtype=tf.float32,
                                  initializer=init_weight)
        bias = tf.get_variable("bias", 
                               shape=[weight_shape[-1]], 
                               dtype=tf.float32,
                               initializer=init_bias)
        act = tf.nn.bias_add(tf.matmul(inputs, weights), bias, name="activation")
        if apply_relu:
            return tf.nn.relu(act, name="relu")
        return act
        


def lrn(inputs, radius, alpha, beta, bias=1.0, name=None):
    """Local Response Normalization
    """
    return tf.nn.local_response_normalization(inputs, 
                                              radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias,
                                              name=name)


def max_pool(inputs, win_size, stride, padding="SAME", name=None):
    """Max Pooling Layer
    """
    return tf.nn.max_pool(inputs, 
                          [1, win_size, win_size, 1],
                          [1, stride, stride, 1],
                          padding,
                          name=name)


def dropout(inputs, keep_prob):
    """Dropout Layer
    """
    return tf.nn.dropout(inputs, keep_prob=keep_prob)