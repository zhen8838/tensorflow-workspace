import tensorflow as tf
import sys
import argparse
import numpy as np
import random
import importlib
from os import environ
import os.path
from datetime import datetime
from utils import *
from mobilenetv1.models import mobilenet_v1
from tqdm import tqdm


def main(args):
    tf.reset_default_graph()
    environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Write arguments to a text file
    write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    # ========= set he seed =============
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.set_random_seed(args.seed)
    # ======== create the dataset =======
    imagesets = get_dataset(args.data_dir)
    image_list, label_list = get_image_paths_and_labels(imagesets)
    # shuffle the datalist
    tmp = list(zip(image_list, label_list))
    random.shuffle(tmp)
    image_list, label_list = zip(*tmp)

    # create the tf.data.dataset
    train_set, step_per_epoch = create_dataset(image_list, label_list,
                                               args.batch_size, args.class_num,
                                               args.image_size, args.image_size, args.seed)
    image_next, label_next = create_dataset_iter(train_set)
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)

    # ======== define the network input nodes ==============
    image_batch = tf.identity(image_next, 'image_batch')
    image_batch = tf.placeholder_with_default(image_batch, shape=[None, args.image_size, args.image_size, 3], name='input')
    label_batch = tf.placeholder_with_default(label_next, shape=[None, args.class_num], name='label_batch')
    print('Number of classes in training set: %d' % len(imagesets))
    print('Number of examples in training set: %d' % len(image_list))

    print('Building training graph')
    # ======= construct the inference graph ===============
    logits, _ = mobilenet_v1.inference(image_batch, args.keep_probability,
                                       phase_train=True, class_num=args.class_num,
                                       weight_decay=args.weight_decay)
    prelogits = logits
    # ======= define the training scalar ======================
    # the dynmic learning_rate
    global_step = tf.train.get_or_create_global_step()
    current_learning_rate = tf.train.exponential_decay(args.init_learning_rate, global_step,
                                                       args.learning_rate_decay_epochs*args.max_nrof_epochs,
                                                       args.learning_rate_decay_factor, staircase=True)

    # Calculate the average cross entropy loss across the batch
    tf.losses.softmax_cross_entropy(label_batch, logits)  # NOTE predict can't be softmax
    total_loss = tf.losses.get_total_loss(name='total_loss')  # NOTE add this can use in test

    # Calc the accuracy
    accuracy, accuracy_op = tf.metrics.accuracy(tf.argmax(label_batch, axis=-1), tf.argmax(logits, axis=-1), name='accuracy')

    # ======= define the training option ======================
    train_op = create_train_op(total_loss, global_step, args.optimizer,
                               current_learning_rate, args.moving_average_decay)

    # Start running operations on the Graph.
    with tf.Session() as sess:
        # init the all varibles~
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # NOTE if custom the model, must run restore after variable initialize
        if os.path.exists(args.pre_train_path):
            # restore the weight
            restore_form_pkl(sess, pklpath=args.pre_train_path, except_last=True)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        tf.summary.scalar('learning rate', current_learning_rate)
        tf.summary.scalar('loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        merged_op = tf.summary.merge_all()
        # ========= start training ===========
        for epoch_cnt in range(args.max_nrof_epochs):
            train_one_epoch(
                sess, step_per_epoch, merged_op, global_step, total_loss, train_op,
                accuracy, accuracy_op, current_learning_rate, summary_writer)


def train_one_epoch(sess, step_per_epoch, merged_op, global_step, loss, train_op,
                    accuracy, accuracy_op, current_learning_rate, writer):

    with tqdm(total=step_per_epoch, bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt}{postfix}]',
              unit=' batch', dynamic_ncols=True) as t:
        for j in range(step_per_epoch):
            merged_op_, loss_, train_op_, learning_rate_, global_step_,  accuracy_, _ = sess.run(
                [merged_op, loss, train_op, current_learning_rate, global_step, accuracy, accuracy_op])
            writer.add_summary(merged_op_, global_step_)
            t.set_postfix(loss='{:<5.3f}'.format(loss_), acc='{:5.2f}%'.format(accuracy_*100), leraning_rate='{:7f}'.format(learning_rate_))
            t.update()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--pre_train_path', type=str,
                        help='Path of pre-train models weight (.pkl) .', default='pretrained/mobilenetv1_1.0.pkl')

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='backup_classifier')

    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.', default='backup_classifier')

    parser.add_argument('--gpus', type=str,
                        help='Indicate the GPUs to be used.', default='0')

    parser.add_argument('--class_num_changed', type=bool, default=False,
                        help='indicate if the class_num is different from pretrained.')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        default='/media/zqh/Datas/DataSet/flower_photos')
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=1)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=32)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=224)
    parser.add_argument('--class_num', type=int,
                        help='Dimensionality of the embedding.', default=5)
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADADELTA')
    parser.add_argument('--init_learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.0006)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=10)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=0.9)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    print('gpu device ID: %s' % args.gpus)
    main(args)
