import tensorflow as tf

inputs = tf.placeholder(dtype=tf.float32, shape=[4, 5, 256, 256, 3])
cell = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=[256, 256, 3], output_channels=96, kernel_shape=[3, 3])
initial_state = cell.zero_state(batch_size=4, dtype=tf.float32)
print(initial_state)
output, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, time_major=False, initial_state=initial_state)
print(output)
print(final_state.h)

