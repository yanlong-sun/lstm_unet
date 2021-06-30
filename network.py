import tensorflow as tf
import ops


class UNet(object):
    def __init__(self, sess, conf, is_train):
        self.sess = sess
        self.conf = conf
        self.is_train = is_train

        self.axis = (2, 3)
        self.channel_axis = 4
        self.input_shape = [conf.batch, conf.depth, conf.height, conf.width, conf.channel]
        self.output_shape = [conf.batch, conf.height, conf.width]
        self.start_slice = conf.start_slice_num

    def inference(self, inputs):
        print('-------------------- LSTM Dense UNet --------------------')
        # -------------------------------------------------- #
        # 1: input
        rate_field = 0
        outputs = inputs[:, 4, :, :, :]
        print('input:            ', outputs.get_shape())

        # -------------------------------------------------- #
        # 2: convolution 1
        name = 'start_block-'
        outputs = ops.conv2d(outputs, rate_field, 96, (3, 3), name + '/conv1', stride=2, is_train=self.is_train, norm=False)
        outputs = ops.conv2d(outputs, rate_field, 96, (3, 3), name + '/conv2', is_train=self.is_train, norm=False)
        conv1 = ops.conv2d(outputs, rate_field, 96, (3, 3), name + '/conv3', is_train=self.is_train, norm=False)

        # == LSTM part +++++++++++++++++++++++++++++++++
        outputs_init = inputs[:, self.start_slice, :, :, :]
        outputs_init = ops.conv2d(outputs_init, rate_field, 96, (3, 3), name + '/conv1', stride=2, is_train=False, norm=False, reuse=True)
        outputs_init = ops.conv2d(outputs_init, rate_field, 96, (3, 3), name + '/conv2', is_train=False, norm=False, reuse=True)
        conv1_lstm = ops.conv2d(outputs_init, rate_field, 96, (3, 3), name + '/conv3', is_train=False, norm=False, reuse=True)
        conv1_lstm = tf.expand_dims(conv1_lstm, 1)
        for i in range(self.start_slice+1, 4):
            outputs_var = inputs[:, i, :, :, :]
            outputs_var = ops.conv2d(outputs_var, rate_field, 96, (3, 3), name + '/conv1', stride=2, is_train=False, norm=False, reuse=True)
            outputs_var = ops.conv2d(outputs_var, rate_field, 96, (3, 3), name + '/conv2', is_train=False, norm=False, reuse=True)
            conv1_var = ops.conv2d(outputs_var, rate_field, 96, (3, 3), name + '/conv3', is_train=False, norm=False, reuse=True)
            conv1_var = tf.expand_dims(conv1_var, 1)
            conv1_lstm = tf.concat([conv1_lstm, conv1_var], 1)
        print('conv1_lstm shape: ', conv1_lstm.get_shape())
        lstm_inputs_0 = conv1_lstm
        cell0 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=[128, 128, 96], output_channels=96, kernel_shape=[3, 3])
        initial_state = cell0.zero_state(batch_size=self.conf.batch, dtype=tf.float32)
        output, final_state = tf.nn.dynamic_rnn(cell0, lstm_inputs_0, dtype=tf.float32, time_major=False, initial_state=initial_state, scope='rnn0')
        conv1 = conv1 + final_state.h
        # == +++++++++++++++++++++++++++++++++++++++++++++

        print('conv1:              ', conv1.get_shape())

        # -------------------------------------------------- #
        # 3: pooling
        outputs = ops.max_pool_2d(conv1, (3, 3), name + '/max_pool')
        print('pooling:            ', outputs.get_shape())

        # -------------------------------------------------- #
        # 4: dense block 1
        name = 'dense_block1-'
        block1 = self.dense_block(outputs, name+'/dense', 6, is_train=self.is_train)

        # == LSTM part +++++++++++++++++++++++++++++++++
        block1_init = conv1_lstm[:, 0, :, :, :]
        block1_init = ops.max_pool_2d(block1_init, (3, 3), name + '/max_pool')
        block1_lstm = self.dense_block(block1_init, name+'/dense', 6, is_train=False, reuse=True)
        block1_lstm = tf.expand_dims(block1_lstm, 1)
        for i in range(1, 4-self.start_slice):
            block1_var = conv1_lstm[:, i, :, :, :]
            block1_var = ops.max_pool_2d(block1_var, (3, 3), name + '/max_pool')
            block1_temp = self.dense_block(block1_var, name+'/dense', 6, is_train=False, reuse=True)
            block1_temp = tf.expand_dims(block1_temp, 1)
            block1_lstm = tf.concat([block1_lstm, block1_temp], 1)
        print('block1_lstm shape: ', block1_lstm.get_shape())
        """
        lstm_inputs_1 = block1_lstm
        cell1 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=[64, 64, 384], output_channels=384, kernel_shape=[3, 3])
        initial_state = cell1.zero_state(batch_size=self.conf.batch, dtype=tf.float32)
        output, final_state = tf.nn.dynamic_rnn(cell1, lstm_inputs_1, dtype=tf.float32, time_major=False, initial_state=initial_state, scope='rnn1')
        block1 = block1 + final_state.h
        # == +++++++++++++++++++++++++++++++++++++++++++++
        """
        print('dense block 1:      ', block1.get_shape())

        # -------------------------------------------------- #
        # 5: transition layer 1
        outputs = ops.conv2d(block1, rate_field, 192, (1, 1), name+'conv11', is_train=self.is_train, bias=False)
        outputs = ops.avg_pool_2d(outputs, (3, 3), name + 'avg_pool')
        print('transition layer 1:  ', outputs.get_shape())

        # -------------------------------------------------- #
        # 6: dense block 2
        name = 'dense_block2-'
        block2 = self.dense_block(outputs, name + '/dense', 12, is_train=self.is_train)

        # == LSTM part +++++++++++++++++++++++++++++++++
        block2_init = block1_lstm[:, 0, :, :, :]
        block2_init = ops.conv2d(block2_init, rate_field, 192, (1, 1), 'dense_block1-conv11', is_train=False, bias=False, reuse=True)
        block2_init = ops.avg_pool_2d(block2_init, (3, 3), name + '/avg_pool')
        block2_lstm = self.dense_block(block2_init, name+'/dense', 12, is_train=False, reuse=True)
        block2_lstm = tf.expand_dims(block2_lstm, 1)
        for i in range(1, 4-self.start_slice):
            block2_var = block1_lstm[:, i, :, :, :]
            block2_var = ops.conv2d(block2_var, rate_field, 192, (1, 1), 'dense_block1-conv11', is_train=False, bias=False, reuse=True)
            block2_var = ops.avg_pool_2d(block2_var, (3, 3), name + '/avg_pool')
            block2_temp = self.dense_block(block2_var, name+'/dense', 12, is_train=False, reuse=True)
            block2_temp = tf.expand_dims(block2_temp, 1)
            block2_lstm = tf.concat([block2_lstm, block2_temp], 1)
        print('block2_lstm shape: ', block2_lstm.get_shape())
        """
        lstm_inputs_2 = block2_lstm
        cell2 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=[32, 32, 768], output_channels=768, kernel_shape=[3, 3])
        initial_state = cell2.zero_state(batch_size=self.conf.batch, dtype=tf.float32)
        output, final_state = tf.nn.dynamic_rnn(cell2, lstm_inputs_2, dtype=tf.float32, time_major=False, initial_state=initial_state, scope='rnn2')
        block2 = block2 + final_state.h
        # == +++++++++++++++++++++++++++++++++++++++++++++
        """
        print('dense block 2:        ', block2.get_shape())

        # -------------------------------------------------- #
        # 7: transition layer 2
        outputs = ops.conv2d(block2, rate_field, 384, (1, 1), name + 'conv11', is_train=self.is_train, bias=False)
        outputs = ops.avg_pool_2d(outputs, (3, 3), name + '/avg_pool')
        print('transition layer 2:    ', outputs.get_shape())

        # -------------------------------------------------- #
        # 8: dense block 3
        name = 'dense_block3-'
        block3 = self.dense_block(outputs, name + '/dense', 36, is_train=self.is_train)

        # == LSTM part +++++++++++++++++++++++++++++++++
        block3_init = block2_lstm[:, 0, :, :, :]
        block3_init = ops.conv2d(block3_init, rate_field, 384, (1, 1), 'dense_block2-conv11', is_train=False, bias=False, reuse=True)
        block3_init = ops.avg_pool_2d(block3_init, (3, 3), name + '/avg_pool')
        block3_lstm = self.dense_block(block3_init, name + '/dense', 36, is_train=False, reuse=True)
        block3_lstm = tf.expand_dims(block3_lstm, 1)
        for i in range(1, 4-self.start_slice):
            block3_var = block2_lstm[:, i, :, :, :]
            block3_var = ops.conv2d(block3_var, rate_field, 384, (1, 1), 'dense_block2-conv11', is_train=False, bias=False, reuse=True)
            block3_var = ops.avg_pool_2d(block3_var, (3, 3), name + '/avg_pool')
            block3_temp = self.dense_block(block3_var, name + '/dense', 36, is_train=False, reuse=True)
            block3_temp = tf.expand_dims(block3_temp, 1)
            block3_lstm = tf.concat([block3_lstm, block3_temp], 1)
        print('block3_lstm shape: ', block3_lstm.get_shape())
        """
        lstm_inputs_3 = block3_lstm
        cell3 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=[16, 16, 2112], output_channels=2112, kernel_shape=[3, 3])
        initial_state = cell3.zero_state(batch_size=self.conf.batch, dtype=tf.float32)
        output, final_state = tf.nn.dynamic_rnn(cell3, lstm_inputs_3, dtype=tf.float32, time_major=False, initial_state=initial_state, scope='rnn3')
        block3 = block3 + final_state.h
        # == +++++++++++++++++++++++++++++++++++++++++++++
        """
        print('dense block 3:          ', block3.get_shape())

        # -------------------------------------------------- #
        # 9: transition layer 3
        outputs = ops.conv2d(block3, rate_field, 1056, (1, 1), name + 'conv11', is_train=self.is_train, bias=False)
        outputs = ops.avg_pool_2d(outputs, (3, 3), name + '/avg_pool')
        print('transition layer 3:     ', outputs.get_shape())

        # -------------------------------------------------- #
        # 10: dense block 4
        name = 'dense_block4-'
        block4 = self.dense_block(outputs, name + '/dense', 24, is_train=self.is_train)
        block4 = ops.conv2d(block4, rate_field, 2112, (1, 1), name + '/conv11', is_train=self.is_train, bias=False)

        # == LSTM part +++++++++++++++++++++++++++++++++
        block4_init = block3_lstm[:, 0, :, :, :]
        block4_init = ops.conv2d(block4_init, rate_field, 1056, (1, 1), 'dense_block3-conv11', is_train=False, bias=False, reuse=True)
        block4_init = ops.avg_pool_2d(block4_init, (3, 3), name + '/avg_pool')
        block4_lstm = self.dense_block(block4_init, name + '/dense', 24, is_train=False, reuse=True)
        block4_lstm = ops.conv2d(block4_lstm, rate_field, 2112, (1, 1), name + '/conv11', is_train=False, bias=False, reuse=True)
        block4_lstm = tf.expand_dims(block4_lstm, 1)
        for i in range(1, 4-self.start_slice):
            block4_var = block3_lstm[:, i, :, :, :]
            block4_var = ops.conv2d(block4_var, rate_field, 1056, (1, 1), 'dense_block3-conv11', is_train=False, bias=False, reuse=True)
            block4_var = ops.avg_pool_2d(block4_var, (3, 3), name + '/avg_pool')
            block4_temp = self.dense_block(block4_var, name + '/dense', 24, is_train=False, reuse=True)
            block4_temp = ops.conv2d(block4_temp, rate_field, 2112, (1, 1), name + '/conv11', is_train=False, bias=False, reuse=True)
            block4_temp = tf.expand_dims(block4_temp, 1)
            block4_lstm = tf.concat([block4_lstm, block4_temp], 1)
        print('block4_lstm shape: ', block4_lstm.get_shape())
        """
        lstm_inputs_4 = block4_lstm
        cell4 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=[8, 8, 2112], output_channels=2112, kernel_shape=[3, 3])
        initial_state = cell4.zero_state(batch_size=self.conf.batch, dtype=tf.float32)
        output, final_state = tf.nn.dynamic_rnn(cell4, lstm_inputs_4, dtype=tf.float32, time_major=False, initial_state=initial_state, scope='rnn4')
        block4 = block4 + final_state.h
        # == +++++++++++++++++++++++++++++++++++++++++++++
        """
        print('dense block 4:           ', block4.get_shape())

        # -------------------------------------------------- #
        # 11: up-sampling layer 1 (with block 3)
        name = 'up1'
        h = 2 * outputs.shape[1]
        w = 2 * outputs.shape[2]
        outputs = tf.image.resize_bilinear(block4, size=(h, w), align_corners=True, name=name+'/bilinear')
        outputs = outputs+block3
        h = 2 * outputs.shape[1]
        w = 2 * outputs.shape[2]
        outputs = ops.conv2d(outputs, rate_field, 768, (3, 3), name+'/conv33', is_train=self.is_train, bias=False)
        print('up-sampling layer 1:       ', outputs.get_shape())
        # == LSTM part +++++++++++++++++++++++++++++++++
        up1_init = block4_lstm[:, 0, :, :, :]
        up1_init = tf.image.resize_bilinear(up1_init, size=(h//2, w//2), align_corners=True, name=name+'/bilinear')
        up1_lstm = up1_init + block3_lstm[:, 0, :, :, :]
        up1_lstm = ops.conv2d(up1_lstm, rate_field, 768, (3, 3), name+'/conv33', is_train=False, bias=False, reuse=True)
        up1_lstm = tf.expand_dims(up1_lstm, 1)
        for i in range(1, 4-self.start_slice):
            up1_var = block4_lstm[:, i, :, :, :]
            up1_var = tf.image.resize_bilinear(up1_var, size=(h//2, w//2), align_corners=True, name=name+'/bilinear')
            up1_temp = up1_var + block3_lstm[:, i, :, :, :]
            up1_temp = ops.conv2d(up1_temp, rate_field, 768, (3, 3), name+'/conv33', is_train=False, bias=False, reuse=True)
            up1_temp = tf.expand_dims(up1_temp, 1)
            up1_lstm = tf.concat([up1_lstm, up1_temp], 1)
        print('up1_lstm shape: ', up1_lstm.get_shape())
        # == +++++++++++++++++++++++++++++++++++++++++++++

        # -------------------------------------------------- #
        # 12: up-sampling layer 2 (with block 2)
        name = 'up2'
        outputs = tf.image.resize_bilinear(outputs, size=(h, w), align_corners=True, name=name+'/bilinear')
        outputs = outputs + block2
        h = 2 * outputs.shape[1]
        w = 2 * outputs.shape[2]
        outputs = ops.conv2d(outputs, rate_field, 384, (3, 3), name+'/conv33', is_train=self.is_train, bias=False)
        print('up-sampling layer 2:        ', outputs.get_shape())
        # == LSTM part +++++++++++++++++++++++++++++++++
        up2_init = up1_lstm[:, 0, :, :, :]
        up2_init = tf.image.resize_bilinear(up2_init, size=(h // 2, w // 2), align_corners=True, name=name + '/bilinear')
        up2_lstm = up2_init + block2_lstm[:, 0, :, :, :]
        up2_lstm = ops.conv2d(up2_lstm, rate_field, 384, (3, 3), name+'/conv33', is_train=False, bias=False, reuse=True)
        up2_lstm = tf.expand_dims(up2_lstm, 1)
        for i in range(1, 4 - self.start_slice):
            up2_var = up1_lstm[:, i, :, :, :]
            up2_var = tf.image.resize_bilinear(up2_var, size=(h // 2, w // 2), align_corners=True, name=name + '/bilinear')
            up2_temp = up2_var + block2_lstm[:, i, :, :, :]
            up2_temp = ops.conv2d(up2_temp, rate_field, 384, (3, 3), name+'/conv33', is_train=False, bias=False, reuse=True)
            up2_temp = tf.expand_dims(up2_temp, 1)
            up2_lstm = tf.concat([up2_lstm, up2_temp], 1)
        print('up2_lstm shape: ', up2_lstm.get_shape())

        # == +++++++++++++++++++++++++++++++++++++++++++++

        # -------------------------------------------------- #
        # 13: up-sampling layer 3 (with block 1)
        name = 'up3'
        outputs = tf.image.resize_bilinear(outputs, size=(h, w), align_corners=True, name=name+'/bilinear')
        outputs = outputs + block1
        h = 2 * outputs.shape[1]
        w = 2 * outputs.shape[2]
        outputs = ops.conv2d(outputs, rate_field, 96, (3, 3), name + '/conv33', is_train=self.is_train, bias=False)
        print('up-sampling layer 3:         ', outputs.get_shape())
        # == LSTM part +++++++++++++++++++++++++++++++++
        up3_init = up2_lstm[:, 0, :, :, :]
        up3_init = tf.image.resize_bilinear(up3_init, size=(h // 2, w // 2), align_corners=True, name=name + '/bilinear')
        up3_lstm = up3_init + block1_lstm[:, 0, :, :, :]
        up3_lstm = ops.conv2d(up3_lstm, rate_field, 96, (3, 3), name + '/conv33', is_train=False, bias=False, reuse=True)
        up3_lstm = tf.expand_dims(up3_lstm, 1)
        for i in range(1, 4 - self.start_slice):
            up3_var = up2_lstm[:, i, :, :, :]
            up3_var = tf.image.resize_bilinear(up3_var, size=(h // 2, w // 2), align_corners=True, name=name + '/bilinear')
            up3_temp = up3_var + block1_lstm[:, i, :, :, :]
            up3_temp = ops.conv2d(up3_temp, rate_field, 96, (3, 3), name + '/conv33', is_train=False, bias=False, reuse=True)
            up3_temp = tf.expand_dims(up3_temp, 1)
            up3_lstm = tf.concat([up3_lstm, up3_temp], 1)
        print('up3_lstm shape: ', up3_lstm.get_shape())

        # == +++++++++++++++++++++++++++++++++++++++++++++
        # 14: up-sampling layer 4 (with conv1)
        name = 'up4'
        outputs = tf.image.resize_bilinear(outputs, size=(h, w), align_corners=True, name=name+'/bilinear')
        outputs = outputs + conv1
        h = 2 * outputs.shape[1]
        w = 2 * outputs.shape[2]
        outputs = ops.conv2d(outputs, rate_field, 96, (3, 3), name + '/conv33', is_train=self.is_train, bias=False)
        print('up-sampling layer 4:          ', outputs.get_shape())
        # == LSTM part +++++++++++++++++++++++++++++++++
        up4_init = up3_lstm[:, 0, :, :, :]
        up4_init = tf.image.resize_bilinear(up4_init, size=(h // 2, w // 2), align_corners=True, name=name + '/bilinear')
        up4_lstm = up4_init + conv1_lstm[:, 0, :, :, :]
        up4_lstm = ops.conv2d(up4_lstm, rate_field, 96, (3, 3), name + '/conv33', is_train=False, bias=False, reuse=True)
        up4_lstm = tf.expand_dims(up4_lstm, 1)
        for i in range(1, 4 - self.start_slice):
            up4_var = up3_lstm[:, i, :, :, :]
            up4_var = tf.image.resize_bilinear(up4_var, size=(h // 2, w // 2), align_corners=True, name=name + '/bilinear')
            up4_temp = up4_var + conv1_lstm[:, i, :, :, :]
            up4_temp = ops.conv2d(up4_temp, rate_field, 96, (3, 3), name + '/conv33', is_train=False, bias=False, reuse=True)
            up4_temp = tf.expand_dims(up4_temp, 1)
            up4_lstm = tf.concat([up4_lstm, up4_temp], 1)
        print('up4_lstm shape: ', up4_lstm.get_shape())
        lstm_inputs_up4 = up4_lstm
        cell_up4 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=[128, 128, 96], output_channels=96, kernel_shape=[3, 3])
        initial_state = cell_up4.zero_state(batch_size=self.conf.batch, dtype=tf.float32)
        output, final_state = tf.nn.dynamic_rnn(cell_up4, lstm_inputs_up4, dtype=tf.float32, time_major=False, initial_state=initial_state, scope='rnn_up4')
        outputs = outputs + final_state.h
        print('up-sampling layer 4 with lstm:          ', outputs.get_shape())
        # == +++++++++++++++++++++++++++++++++++++++++++++

        # -------------------------------------------------- #
        # 15: up-sampling layer 5
        name = 'up5'
        outputs = tf.image.resize_bilinear(outputs, size=(h, w), align_corners=True, name=name+'/bilinear')

        # branch 1
        branch1 = ops.conv2d(outputs, rate_field, 32, (3, 3), name+'/branch1-1', is_train=self.is_train, bias=False)
        branch1 = ops.conv2d(branch1, rate_field, 32, (3, 3), name+'/branch1-2', is_train=self.is_train, bias=False)
        branch1 = ops.conv2d(branch1, rate_field, 1, (1, 1), name+'/branch1-3', is_train=self.is_train, bias=False)

        # branch 2
        branch2 = ops.conv2d(outputs, rate_field, 32, (3, 3), name+'/branch2-1', is_train=self.is_train, bias=False)
        branch2 = ops.conv2d(branch2, rate_field, 32, (3, 3), name+'/branch2-2', is_train=self.is_train, bias=False)
        branch2 = ops.conv2d(branch2, rate_field, 1, (1, 1), name+'/branch2-3', is_train=self.is_train, bias=False)

        # branch 3
        branch3 = ops.conv2d(outputs, rate_field, 32, (3, 3), name+'/branch3-1', is_train=self.is_train, bias=False)
        branch3 = ops.conv2d(branch3, rate_field, 32, (3, 3), name+'/branch3-2', is_train=self.is_train, bias=False)
        branch3 = ops.conv2d(branch3, rate_field, 2, (1, 1), name+'/branch3-3', is_train=self.is_train, bias=False)

        outputs = tf.concat([branch1, branch2, branch3], 3, name=name+'concat')
        print('up-sampling layer 5:             ', outputs.get_shape())

        return outputs, rate_field

    # DENSE BLOCK
    def dense_block(self, inputs, name, num, is_train=True, reuse=False):
        rate_field = inputs
        for i in range(0, num):
            outputs = ops.conv2d(inputs, rate_field, 192, (1, 1), name + '/conv11_' + str(i+1), is_train=is_train, bias=False, reuse=reuse)
            outputs = ops.conv2d(outputs, rate_field, 48, (3, 3), name + '/conv33_' + str(i+1), is_train=is_train, bias=False, reuse=reuse)
            inputs = tf.concat([inputs, outputs], 3, name=name+'concat'+str(i+1))
        return inputs
