import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
_MOMENTUM = 0.9


def batch_norm_relu(inputs, is_training, data_format):
    """Performs a batch normalization followed by a ReLU."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=is_training, fused=True)
    inputs = tf.nn.relu(inputs)
    return inputs


def conv2d_bn_relu(inputs, filters, kernel_size, strides,
                   is_training, data_format):
    inputs = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        data_format=data_format)
    inputs = batch_norm_relu(inputs, is_training, data_format)
    return inputs
