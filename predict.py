import os
import time
import sys
import tensorflow as tf
import numpy as np
import argparse
from scipy import misc
from utils import *
from skimage.io import imshow, show


def main(args):
    g = tf.get_default_graph()
    sess = tf.Session()
    load_pb_model(args.model_path)
    input_image = g.get_tensor_by_name('inputs:0')
    logits = g.get_tensor_by_name('MobileNetV1/Bottleneck2/BatchNorm/Reshape_1:0')
    predict = tf.nn.softmax(logits)

    image_path_list = [os.path.join(args.image_dir, name) for name in os.listdir(os.path.join(os.getcwd(), args.image_dir)) if '.jpg' in name]
    for image_path in image_path_list:
        resized_img, one_hot_label = parser(image_path, 0, args.class_num, args.image_size, args.image_size, False)
        resized_img = tf.expand_dims(resized_img, 0)
        img, label = sess.run([resized_img, one_hot_label])
        logit, pred = sess.run([logits, predict], feed_dict={input_image: img})
        pred = np.squeeze(pred)
        des_idx = np.argsort(pred)

        with open("data/names.list", "r") as f:
            lines = f.readlines()
        print('预测 {} :'.format(image_path))
        for j in range(5):
            print("%.2f%%--%s" % (pred[des_idx[args.class_num-1-j]]*100, lines[des_idx[args.class_num-1-j]].strip()))
        print()

#  inputs={'Input_image': batch_image, 'Input_label': batch_label},
#    outputs={'Output_label': predict}


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.', default='')
    parser.add_argument('--image_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        default='data')
    parser.add_argument('--image_size', type=int,
                        help='image size.', default=224)
    parser.add_argument('--class_num', type=int,
                        help='Dimensionality of the embedding.', default=1000)

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
