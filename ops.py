import tensorflow as tf


def conv2d(inputs, rate_field, num_outputs, kernel_size, scope, stride=1, rate=1,
           is_train=True, bias=True, norm=True, activation=True, d_format='NHWC'):
    # bias
    if bias:
        outputs = tf.contrib.layers.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                                           data_format=d_format, rate=rate, activation_fn=None, scope=scope)
    else:
        outputs = tf.contrib.layers.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                                           data_format=d_format, rate=rate, activation_fn=None, biases_initializer=None,
                                           scope=scope)

    # BN
    if norm:
        outputs = tf.contrib.layers.batch_norm(outputs, decay=0.9, center=True, scale=True, activation_fn=None,
                                               epsilon=1e-5, is_training=is_train, scope=scope + '/batch_norm',
                                               data_format=d_format)

    if activation:
        outputs = tf.nn.relu(outputs, name=scope + '/relu')

    return outputs


def max_pool_2d(inputs, kernel_size, scope, stride=2, padding='SAME', data_format='NHWC'):
    outputs = tf.contrib.layers.max_pool2d(inputs, kernel_size, stride=stride,
                                           scope=scope + '/max_pool', padding=padding, data_format=data_format)

    return outputs


def avg_pool_2d(inputs, kernel_size, scope, stride=2, padding='SAME', data_format='NHWC'):
    outputs = tf.contrib.layers.avg_pool2d(inputs, kernel_size, stride=stride,
                                           scope=scope + '/avg_pool', padding=padding, data_format=data_format)

    return outputs
