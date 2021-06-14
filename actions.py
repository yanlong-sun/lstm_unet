import tensorflow as tf
from data_reader import H5DataLoader
import numpy as np
import time
import os
from network import UNet
from ccv import CCV
from img_utils import imsave
import random
random.seed(7)


def dice_coeff(gt, out):
    intersection = np.sum(gt * out)
    dice = (2. * intersection + 0.000001) / (np.sum(gt) + np.sum(out) + 0.000001)
    if dice > 1:
        return -1
    return dice


class Actions(object):
    def __init__(self, sess, conf):
        # step 1: init
        self.sess = sess
        self.conf = conf
        self.batch_axis = 0
        self.axis = (2, 3)
        self.channel_axis = 4
        self.input_shape = [conf.batch, conf.depth, conf.height, conf.width, conf.channel]
        self.output_shape = [conf.batch, conf.height, conf.width]

        # step 2: mkdir
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        if not os.path.exists(conf.record_dir):
            os.makedirs(conf.record_dir)
        if not os.path.exists(conf.pred_dir):
            os.makedirs(conf.pred_dir)

        # step 3: configure network

        # input/output shape
        self.inputs = tf.placeholder(tf.float32, self.input_shape, name='inputs')
        self.annotations = tf.placeholder(tf.int64, self.output_shape, name='annotations')
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        expanded_annotations = tf.expand_dims(self.annotations, -1, name='annotations/expand_dims')
        one_hot_annotations = tf.squeeze(expanded_annotations, axis=[3], name='annotations/squeeze')
        one_hot_annotations = tf.one_hot(one_hot_annotations, depth=2, axis=3, name='annotations/one_hot')

        # load network
        model = UNet(self.sess, self.conf, self.is_train)
        self.outputs, self.rates = model.inference(self.inputs)

        ###
        shape1 = one_hot_annotations.shape
        shape2 = self.outputs.shape

        if shape1[1].value != shape2[1].value or shape1[2].value != shape2[2].value or shape1[3].value != shape2[3].value:
            print('shape of one_hot_annotations: ', shape1)
            print('shape of outputs: ', shape2)

        # loss
        self.net_pred = self.outputs[:, :, :, 2:]
        self.decoded_net_pred = tf.argmax(self.net_pred, 3, name='accuracy/decode_net_pred')
        losses1 = tf.losses.softmax_cross_entropy(one_hot_annotations, self.net_pred, scope='loss/losses1')
        self.predicted_prob = tf.nn.softmax(self.net_pred, name='softmax')

        # CCV
        self.pred = CCV(self.outputs, self.inputs[:, 4, :, :, :], 2, 0.5, 1e-8)

        lambda1 = 0.01
        self.pred = tf.squeeze(self.pred)
        losses2 = tf.reduce_sum(tf.square(self.pred - tf.cast(self.annotations, "float32")))
        losses = lambda1 * losses1 + losses2
        # optimize
        self.loss_op = tf.reduce_mean(losses, name='loss/loss_op')
        optimizer = tf.train.AdamOptimizer(learning_rate=self.conf.learning_rate,
                                           beta1=self.conf.beta1,
                                           beta2=self.conf.beta2,
                                           epsilon=self.conf.epsilon)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss_op, name='train_op')

        # get results
        self.predictions = self.pred
        high0 = tf.ones(self.annotations.shape, "int64")
        low0 = tf.zeros(self.annotations.shape, "int64")
        gamma0 = tf.ones(self.annotations.shape) * 0.5
        self.decoded_predictions = tf.where(tf.greater_equal(self.predictions, gamma0), high0, low0)
        correct_prediction = tf.equal(self.annotations, self.decoded_predictions, name='accuracy/correct_pred')
        self.accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32, name='accuracy/cast'),
                                          name='accuracy/accuracy_op')
        self.out = tf.cast(self.decoded_predictions, tf.float32)
        self.gt = tf.cast(self.annotations, tf.float32)

        #
        tf.set_random_seed(int(time.time()))
        self.sess.run(tf.global_variables_initializer())

        trainable_vars = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'batch_norm/moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'batch_norm/moving_variance' in g.name]
        trainable_vars += bn_moving_vars
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)

    def train(self):
        valid_loss_list = []
        valid_dice_list = []
        valid_acc_list = []
        train_loss_list = []
        train_dice_list = []
        train_acc_list = []

        train_reader = H5DataLoader(self.conf.data_dir + self.conf.train_data)
        valid_reader = H5DataLoader(self.conf.data_dir + self.conf.valid_data)
        self.sess.run(tf.local_variables_initializer())
        for epoch_num in range(self.conf.max_epoch):
            if epoch_num % 10 == 1:
                print('training step: ' + str(epoch_num) + '/' + str(self.conf.max_epoch))
            if epoch_num % self.conf.valid_step == 1:
                # save record
                # valid
                inputs, annotations = valid_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs, self.annotations: annotations[:, 4, :, :], self.is_train: False}
                loss, accuracy = self.sess.run([self.loss_op, self.accuracy_op], feed_dict=feed_dict)
                out, gt = self.sess.run([self.out, self.gt], feed_dict=feed_dict)
                dice = dice_coeff(gt, out)
                print('training step: ' + str(epoch_num) + '/' + str(self.conf.max_epoch))
                print('valid...')
                print(epoch_num, '------valid loss: ', loss)
                print(epoch_num, '------valid acc : ', accuracy)
                print(epoch_num, '------valid dice: ', dice)
                # loss
                valid_loss_list.append(loss)
                np.save(self.conf.record_dir + 'valid_loss.npy', np.array(valid_loss_list))
                # dice
                valid_dice_list.append(dice)
                np.save(self.conf.record_dir + 'valid_dice.npy', np.array(valid_dice_list))
                # accuracy
                valid_acc_list.append(accuracy)
                np.save(self.conf.record_dir + 'valid_acc.npy', np.array(valid_acc_list))

                # train
                inputs, annotations = train_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs, self.annotations: annotations[:, 4, :, :], self.is_train: True}
                _, loss, accuracy = self.sess.run([self.train_op, self.loss_op, self.accuracy_op], feed_dict=feed_dict)
                out, gt = self.sess.run([self.out, self.gt], feed_dict=feed_dict)
                dice = dice_coeff(gt, out)
                print('train...')
                print(epoch_num, '------train loss: ', loss)
                print(epoch_num, '------train acc : ', accuracy)
                print(epoch_num, '------train dice: ', dice)
                # loss
                train_loss_list.append(loss)
                np.save(self.conf.record_dir + 'train_loss.npy', np.array(train_loss_list))
                # dice
                train_dice_list.append(dice)
                np.save(self.conf.record_dir + 'train_dice.npy', np.array(train_dice_list))
                # accuracy
                train_acc_list.append(accuracy)
                np.save(self.conf.record_dir + 'train_acc.npy', np.array(train_acc_list))
            else:
                inputs, annotations = train_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs, self.annotations: annotations[:, 4, :, :], self.is_train: True}
                _, loss = self.sess.run([self.train_op, self.loss_op], feed_dict=feed_dict)

            if epoch_num % self.conf.save_step == 1:
                self.save(epoch_num)

    def predict(self):
        predictions = []
        count = 0

        print('predicting......', self.conf.test_epoch)
        if self.conf.test_epoch > 0:
            self.reload(self.conf.test_epoch)
        else:
            print("please set a reasonable test_epoch")
            return
        test_reader = H5DataLoader(self.conf.data_dir + self.conf.test_data, False)
        self.sess.run(tf.local_variables_initializer())

        while True:
            inputs, annotations = test_reader.next_batch(self.conf.batch)
            if inputs.shape[0] < self.conf.batch:
                break

            feed_dict = {self.inputs: inputs, self.annotations: annotations[:, 4, :, :], self.is_train: False}
            pred_result = self.sess.run(self.decoded_net_pred, feed_dict=feed_dict)
            pred_result = pred_result

            predictions.append(pred_result)

            count += 1
            if count == self.conf.test_num:
                break

        print('saving predictions...')
        print(np.shape(predictions))
        num = 0
        for index, prediction in enumerate(predictions):
            for i in range(prediction.shape[0]):
                num += 1
                pred_result_temp = prediction[i]
                imsave(pred_result_temp, self.conf.pred_dir + str(index * prediction.shape[0] + i) + '.png')  #<--------------------

    def save(self, step):
        checkpoint_path = os.path.join(self.conf.model_dir, 'model')
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(self.conf.model_dir, 'model')
        model_path = checkpoint_path + '-' + str(step)
        if not os.path.exists(model_path + '.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)

