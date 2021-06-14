import tensorflow as tf
import argparse
import os
from actions import Actions


def configure():
    flags = tf.app.flags

    flags.DEFINE_integer('batch', 4, 'batch_size')
    flags.DEFINE_integer('height', 256, 'height of the slices')
    flags.DEFINE_integer('width', 256, 'width of the slices')
    flags.DEFINE_integer('depth', 5, 'depth of the slices')
    flags.DEFINE_integer('channel', 3, 'channel')
    flags.DEFINE_integer('start_slice_num', 0, 'start_slice')

    flags.DEFINE_integer('test_num', 3039, 'number of test slices')

    flags.DEFINE_string('logdir', '../network/log/', 'path to log of model')
    flags.DEFINE_string('model_dir', '../network/model/', 'path to model')
    flags.DEFINE_string('record_dir', '../network/record/', 'path to record')

    flags.DEFINE_string('data_dir', '../Dataset/h5py/', 'path to dataset')
    flags.DEFINE_string('pred_dir', '../predictions/', 'path to predictions')
    flags.DEFINE_string('train_data', 'training_data.hdf5', 'training data')
    flags.DEFINE_string('valid_data', 'valid_data.hdf5', 'valid data')
    flags.DEFINE_string('test_data', 'test_data.hdf5', 'test data')

    flags.DEFINE_integer('max_epoch', 30001, 'num of epoch')
    flags.DEFINE_integer('test_epoch', 29001, 'choose epoch for testing')
    flags.DEFINE_integer('valid_step', 1000, 'step to valid')
    flags.DEFINE_integer('save_step', 1000, 'step to save')

    flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    flags.DEFINE_float('beta1', 0.9, 'beta1')
    flags.DEFINE_float('beta2', 0.99, 'beta2')
    flags.DEFINE_float('epsilon', 1e-8, 'epsilon')

    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


def train():
    model = Actions(sess, configure())
    model.train()


def predict():
    model = Actions(sess, configure())
    model.predict()


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', dest='action', type=str, default='train', help='train or predict')
    args = parser.parse_args()
    if args.action not in ['train', 'predict']:
        print('invalid action: ', args.action)
        print('Please input a action: train or predict')
    elif args.action == 'predict':
        predict()
    else:
        train()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.app.run()
