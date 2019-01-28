import tensorflow as tf
import os
import re
import sys
import argparse
from models import mobilenet_v1
from tensorflow.python.framework import graph_util


def freeze_graph(input_dir, output_file, class_num):
    graph = tf.get_default_graph()  # 获得默认的图
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(input_dir)

    images_placeholder = tf.placeholder(tf.float32, shape=(1, 224, 224, 3), name='inputs')

    logits, _ = mobilenet_v1.inference(images_placeholder, keep_probability=0,
                                       phase_train=False, class_num=class_num)

    saver = tf.train.Saver(tf.global_variables())

    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    output_node_names = "MobileNetV1/Bottleneck2/BatchNorm/Reshape_1"

    with tf.Session() as sess:

        saver.restore(sess, ckpt.model_checkpoint_path)  # 恢复图并得到数据

        # fix batch norm nodes
        for node in input_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr:
                    del node.attr['use_locking']

        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开
        with tf.gfile.GFile(output_file, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点         # for op in graph.get_operations():        #     print(op.name, op.values())


def main(args):
    freeze_graph(args.ckpt_dir, args.output_file, args.class_num)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_num', type=int,
                        help='Dimensionality of the embedding.', default=1000)
    parser.add_argument('ckpt_dir', type=str,
                        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')
    parser.add_argument('output_file', type=str,
                        help='Filename for the exported graphdef protobuf (.pb)')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
