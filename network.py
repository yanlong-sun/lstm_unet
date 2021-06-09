import tensorflow as tf
import ops


class UNet(object):
    def __init__(self, sess, conf, is_train):
        self.sess = sess
        self.conf = conf
        self.is_train = is_train

        self.data_format = 'NHWC'
        self.axis = (1, 2, 3)
        self.channel_axis = 4
        self.input_shape = [conf.batch, conf.height, conf.width, conf.depth, conf.channel]
        self.output_shape = [conf.batch, conf.height, conf.width]

    def inference(self, inputs):
        print('-------------------- LSTM Dense UNet --------------------')
        # -------------------------------------------------- #
        # 1: input
        rate_field = 0
        outputs = inputs[:, :, :, 1, :]
        print('input:            ', outputs.get_shape())

        # -------------------------------------------------- #
        # 2: convolution 1
        name = 'start_block'
        outputs = ops.conv2d(outputs, rate_field, 96, (3, 3), name + '/conv1', stride=2, is_train=self.is_train, norm=False)
        outputs = ops.conv2d(outputs, rate_field, 96, (3, 3), name + '/conv2', is_train=self.is_train, norm=False)
        conv1 = ops.conv2d(outputs, rate_field, 96, (3, 3), name + '/conv3', is_train=self.is_train, norm=False)
        print('conv1:              ', conv1.get_shape())

        # -------------------------------------------------- #
        # 3: pooling
        outputs = ops.max_pool_2d(conv1, (3, 3), name + '/max_pool')
        print('pooling:            ', outputs.get_shape())

        # -------------------------------------------------- #
        # 4: dense block 1
        name = 'dense_block1'
        block1 = self.dense_block(outputs, name+'/dense', 6)
        print('dense block 1:      ', block1.get_shape())

        # -------------------------------------------------- #
        # 5: transition layer 1
        outputs = ops.conv2d(block1, rate_field, 192, (1, 1), name+'conv11', is_train=self.is_train, bias=False)
        outputs = ops.avg_pool_2d(outputs, (3, 3), name + 'avg_pool')
        print('transition layer 1:  ', outputs.get_shape())

        # -------------------------------------------------- #
        # 6: dense block 2
        name = 'dense_block2'
        block2 = self.dense_block(outputs, name + '/dense', 12)
        print('dense block 2:        ', block2.get_shape())

        # -------------------------------------------------- #
        # 7: transition layer 2
        outputs = ops.conv2d(block2, rate_field, 384, (1, 1), name + 'conv11', is_train=self.is_train, bias=False)
        outputs = ops.avg_pool_2d(outputs, (3, 3), name + '/avg_pool')
        print('transition layer 2:    ', outputs.get_shape())

        # -------------------------------------------------- #
        # 8: dense block 3
        name = 'dense_block3'
        block3 = self.dense_block(outputs, name + '/dense', 36)
        print('dense block 3:          ', block3.get_shape())

        # -------------------------------------------------- #
        # 9: transition layer 3
        outputs = ops.conv2d(block3, rate_field, 1056, (1, 1), name + 'conv11', is_train=self.is_train, bias=False)
        outputs = ops.avg_pool_2d(outputs, (3, 3), name + '/avg_pool')
        print('transition layer 3:     ', outputs.get_shape())

        # -------------------------------------------------- #
        # 10: dense block 4
        name = 'dense_block4'
        block4 = self.dense_block(outputs, name + '/dense', 24)
        block4 = ops.conv2d(block4, rate_field, 2112, (1, 1), name + '/conv11', is_train=self.is_train, bias=False)
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

        # -------------------------------------------------- #
        # 12: up-sampling layer 2 (with block 2)
        name = 'up2'
        outputs = tf.image.resize_bilinear(outputs, size=(h, w), align_corners=True, name=name+'/bilinear')
        outputs = outputs + block2
        h = 2 * outputs.shape[1]
        w = 2 * outputs.shape[2]
        outputs = ops.conv2d(outputs, rate_field, 384, (3, 3), name+'/conv33', is_train=self.is_train, bias=False)
        print('up-sampling layer 2:        ', outputs.get_shape())

        # -------------------------------------------------- #
        # 13: up-sampling layer 3 (with block 1)
        name = 'up3'
        outputs = tf.image.resize_bilinear(outputs, size=(h, w), align_corners=True, name=name+'/bilinear')
        outputs = outputs + block1
        h = 2 * outputs.shape[1]
        w = 2 * outputs.shape[2]
        outputs = ops.conv2d(outputs, rate_field, 96, (3, 3), name + '/conv33', is_train=self.is_train, bias=False)
        print('up-sampling layer 3:         ', outputs.get_shape())

        # -------------------------------------------------- #
        # 14: up-sampling layer 4 (with conv1)
        name = 'up4'
        outputs = tf.image.resize_bilinear(outputs, size=(h, w), align_corners=True, name=name+'/bilinear')
        outputs = outputs + conv1
        h = 2 * outputs.shape[1]
        w = 2 * outputs.shape[2]
        outputs = ops.conv2d(outputs, rate_field, 96, (3, 3), name + '/conv33', is_train=self.is_train, bias=False)
        print('up-sampling layer 4:          ', outputs.get_shape())

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

        outputs = tf.concat([branch1, branch2, branch3], self.channel_axis, name=name+'concat')
        print('up-sampling layer 5:             ', outputs.get_shape())

        return outputs, rate_field

    # DENSE BLOCK
    def dense_block(self, inputs, name, num):
        rate_field = inputs
        for i in range(num):
            outputs = ops.conv2d(inputs, rate_field, 192, (1, 1), name+'/conv11_'+str(i+1), is_train=self.is_train, bias=False)
            outputs = ops.conv2d(outputs, rate_field, 48, (3, 3), name+'/conv33_'+str(i+1), is_train=self.is_train, bias=False)
            inputs = tf.concat([inputs, outputs], self.channel_axis, name=name+'concat'+str(i+1))
        return inputs
