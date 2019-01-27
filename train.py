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
    network = mobilenet_v1
    image_size = (args.image_size, args.image_size)

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
    image_batch, label_batch = create_dataset_iter(train_set)
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)

    tf.set_random_seed(args.seed)
    global_step = tf.train.create_global_step()
    # ======== control the hyperparamter ========
    learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
    batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')
    labels_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='labels')
    control_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='control')
    # ======== define the network input nodes ==============
    image_batch = tf.identity(image_batch, 'image_batch')
    image_batch = tf.identity(image_batch, 'input')
    label_batch = tf.identity(label_batch, 'label_batch')
    print('Number of classes in training set: %d' % len(imagesets))
    print('Number of examples in training set: %d' % len(image_list))

    print('Building training graph')
    # ======= construct the inference graph ===============
    logits, _ = network.inference(image_batch, args.keep_probability,
                                  phase_train=phase_train_placeholder, class_num=args.class_num,
                                  weight_decay=args.weight_decay)
    prelogits = logits
    # ======= define the training scalar ======================
    # the dynmic learning_rate
    learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                               args.learning_rate_decay_epochs*step_per_epoch,
                                               args.learning_rate_decay_factor, staircase=True)
    tf.summary.scalar('learning rate', learning_rate)

    # Calculate the average cross entropy loss across the batch
    tf.losses.softmax_cross_entropy(label_batch, logits)  # NOTE predict can't be softmax
    total_loss = tf.losses.get_total_loss(name='total_loss')  # NOTE add this can use in test
    tf.summary.scalar('loss', total_loss)
    # Calc the accuracy
    accuracy, accuracy_op = tf.metrics.accuracy(labels=label_batch, predictions=logits, name='accuracy')
    tf.summary.scalar('accuracy', accuracy)
    # ======= define the training option ======================
    train_op = create_train_op(total_loss, global_step, args.optimizer,
                               learning_rate, args.moving_average_decay,
                               tf.global_variables(), args.log_histograms)

    # Create a saver
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    # var_list += bn_moving_vars
    var_list = list(set(var_list+bn_moving_vars))

    saver = tf.train.Saver(var_list=var_list, max_to_keep=10)

    # Start running operations on the Graph.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    merged_op = tf.summary.merge_all()
    # ========= start training ===========
    with sess.as_default():
        # total_steps = args.max_nrof_epochs*step_per_epoch
        print('Running training')
        # todo add train function
        for epoch_cnt in range(1, args.max_nrof_epochs+1):
            cont = train_one_epoch(
                args, sess, epoch_cnt, step_per_epoch, merged_op,
                learning_rate_placeholder, phase_train_placeholder,
                global_step, total_loss, train_op, args.learning_rate_schedule_file,
                accuracy, accuracy_op, learning_rate, prelogits, summary_writer)


def train_one_epoch(args, sess, epoch_cnt, step_per_epoch, merged_op, learning_rate_placeholder, phase_train_placeholder,
                    global_step, loss, train_op, learning_rate_schedule_file,
                    accuracy, accuracy_op, learning_rate,
                    prelogits, writer):
    # when args.learning_rate is -1 ,the learning_rate read from file
    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        lr = get_learning_rate_from_file(learning_rate_schedule_file, epoch_cnt)
    assert lr >= 0

    with tqdm(total=step_per_epoch,
              bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt}{postfix}]',
              unit=' batch', dynamic_ncols=True) as t:
        for j in range(step_per_epoch):
            merged_op_, loss_, train_op_, global_step_, prelogits_, learning_rate_, accuracy_, _ = sess.run(
                [merged_op, loss, train_op, global_step, prelogits, learning_rate, accuracy, accuracy_op],
                feed_dict={learning_rate_placeholder: lr, phase_train_placeholder: True})
            writer.add_summary(merged_op_, global_step_)
            t.set_postfix(loss='{:<5.3f}'.format(loss_), acc='{:5.2f}%'.format(accuracy_*100), leraning_rate='{:7f}'.format(learning_rate_))
            t.update()
    return True


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='backup_classifier')

    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.', default='backup_classifier')

    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)

    parser.add_argument('--gpus', type=str,
                        help='Indicate the GPUs to be used.', default='2')

    parser.add_argument('--class_num_changed', type=bool, default=False,
                        help='indicate if the class_num is different from pretrained.')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        default='/media/zqh/Datas/DataSet/flower_photos')
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=5)
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
                        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.0005)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=10)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)

    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')

    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    print('gpu device ID: %s' % args.gpus)
    main(args)
