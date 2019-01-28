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
from models import mobilenet_v1
from tqdm import tqdm


def main(args):
    g = tf.get_default_graph()
    # create the dir
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)

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

    train_set, step_per_epoch = create_dataset(image_list, label_list,  args.batch_size, args.class_num,
                                               args.image_size, args.image_size, args.seed)
    image_next, label_next = create_dataset_iter(train_set)
    # ========= printf log ==============
    print('Log directory: %s' % log_dir)
    print('Number of classes in training set: %d' % len(imagesets))
    print('Number of examples in training set: %d' % len(image_list))

    # ========= construct the network ========
    # NOTE add placeholder_with_default node for test
    batch_image = tf.placeholder_with_default(image_next, shape=[None, 224, 224, 3], name='Input_image')
    batch_label = tf.placeholder_with_default(label_next, shape=[None, 5], name='Input_label')
    logits, _ = mobilenet_v1.inference(batch_image, args.keep_probability, phase_train=True,
                                       class_num=args.class_num, weight_decay=args.weight_decay)

    predict = tf.identity(logits, name='Output_label')
    # ========= define loss ==================
    tf.losses.softmax_cross_entropy(batch_label, predict)  # NOTE predict can't be softmax
    total_loss = tf.losses.get_total_loss(name='total_loss')  # NOTE add this can use in test
    # ========= get global steps =============
    global_step = tf.train.create_global_step()

    # ========= define train optimizer =======
    current_learning_rate = tf.train.exponential_decay(args.init_learning_rate, global_step,
                                                       args.learning_rate_decay_epochs*args.max_nrof_epochs,
                                                       args.learning_rate_decay_factor, staircase=False)

    train_op = create_train_op(total_loss, global_step, args.optimizer, current_learning_rate)
    # ========= calc the accuracy ============
    accuracy, accuracy_op = tf.metrics.accuracy(tf.argmax(batch_label, axis=-1), tf.argmax(predict, axis=-1), name='clac_acc')

    with tf.Session() as sess:
        # init the model and restore the pre-train weight
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # NOTE the accuracy must init local variable
        if args.pretrained_model == None:
            pass
        elif os.path.isdir(args.pretrained_model):
            print('Restoring pretrained model: %s' % args.pretrained_model)
            saver_restore = tf.train.Saver()
            saver_restore.restore(sess, tf.train.latest_checkpoint(args.pretrained_model))
            print('load pre-train weight success!')
        elif '.pkl' in args.pretrained_model and os.path.exists(args.pretrained_model):  # load pb model
            restore_form_pkl(sess, pklpath=args.pretrained_model, except_last=True)
            print('load pre-train weight success!')

        # define the log and saver
        writer = tf.summary.FileWriter(log_dir, graph=g)
        tf.summary.scalar('loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('leraning rate', current_learning_rate)
        merged = tf.summary.merge_all()
        # start training
        for i in range(args.max_nrof_epochs):
            with tqdm(total=step_per_epoch, bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt}{postfix}]', unit=' batch', dynamic_ncols=True) as t:
                for j in range(step_per_epoch):
                    summary, _, losses, acc, _, lrate, step_cnt = sess.run([merged, train_op, total_loss, accuracy, accuracy_op,
                                                                            current_learning_rate, global_step])
                    writer.add_summary(summary, step_cnt)
                    t.set_postfix(loss='{:<5.3f}'.format(losses), acc='{:5.2f}%'.format(acc*100), leraning_rate='{:7f}'.format(lrate))
                    t.update()
        tf.saved_model.simple_save(sess, os.path.join(log_dir, 'backup'),
                                   inputs={'Input_image': batch_image, 'Input_label': batch_label},
                                   outputs={'Output_label': predict})


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts. now support the ckpt dir and .pkl file')

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='backup_classifier')

    parser.add_argument('--gpus', type=str,
                        help='Indicate the GPUs to be used.', default='0')

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

    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    print('gpu device ID: %s' % args.gpus)
    main(args)
