import sys
import tensorflow as tf
from mobilenetv1.models import mobilenet_v1
from utils import *
import numpy as np
import argparse


def test_two_node(g, sess):
    """ 测试两种节点输出的值是否相同 """
    we = g.get_tensor_by_name('MobileNetV1/Conv2d_12_1x1/weights:0')
    re = g.get_tensor_by_name('MobileNetV1/Conv2d_12_1x1/weights/read:0')
    r = sess.run(re)
    w = sess.run(we)
    assert r.any() == w.any()


def check_var_list():
    pboptlist = get_optlist_from_pb(PB_FILE_PATH)
    pbvarlist = get_vars_from_optlist(pboptlist)

    tf.reset_default_graph()
    g = tf.Graph
    image = tf.placeholder(tf.float32, shape=(1, 224, 224, 3), name='Input')
    nets, _ = mobilenet_v1.inference(image, 1.0)
    modelvarlist = tf.global_variables()  # ! 获得所有变量
    # ! 此模型一共112个变量,判断变量数相同
    assert len(modelvarlist) == len(pbvarlist)
    run_assign_list = []
    for i, var in enumerate(modelvarlist):
        assert var.name[:-2] == pbvarlist[i].name[:-5]


def get_optlist_from_pb(g, sess, filepath)->list:
    """ 从pb文件获得所有操作节点 """
    with tf.gfile.GFile(filepath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # 把pb文件的图导入当前图
        tf.import_graph_def(graph_def, name='')
    optlist = g.get_operations()  # 获得所有节点
    return optlist


def get_vars_from_optlist(optlist: list)->list:
    """ 从optlist获得所有的变量节点 """
    varlist = [node for node in optlist if '/read' in node.name]
    return varlist


def convert_vars_to_tensor(g, varlist: list)->list:
    """ 把varlist中的操作转变为可运行的tensor """
    tensorlist = []
    for var in varlist:
        tensorlist.append(g.get_tensor_by_name(var.name+':0'))
    return tensorlist


def pb_2_pkl(PB_FILE_PATH, PKL_FILE_PATH, LIST_FILE_PATH):
    """ 运行此程序将pb文件转化成pkl文件 """
    g = tf.get_default_graph()  # type:tf.Graph
    sess = tf.Session()
    print("load graph")
    # 加载pb文件,获取所有操作符
    optlist = get_optlist_from_pb(g, sess, PB_FILE_PATH)
    # 取出读取变量操作符
    varlist = get_vars_from_optlist(optlist)
    # 获得变量tensor
    tensorlist = convert_vars_to_tensor(g, varlist)
    assert len(tensorlist) == 112
    # 将所有变量存入字典
    vardict = {}
    for v in tensorlist:
        vardict[v.name.replace('/read', '')] = sess.run(v)

    save_pkl(vardict, PKL_FILE_PATH)
    # 写入名字,并写入32个值做对比
    with open(LIST_FILE_PATH, 'w') as f:
        for k, v in vardict.items():
            f.write(k+','+'{}\n'.format(list(np.ravel(v)[:32])))


def main(args):
    pb_2_pkl(args.pb_path, args.pkl_path, args.pkl_path[:-4]+'_node.csv')


def parse_arguments(argv):
    parser = argparse.ArgumentParser(usage='Convert the pb file to pkl file')
    parser.add_argument('--pb_path', type=str,
                        help='Tensorflow model (.pb) file path', default='pretrained/mobilenetv1_1.0.pb')
    parser.add_argument('--pkl_path', type=str,
                        help='Directory weight file (.pkl) file path', default='pretrained/mobilenetv1_1.0.pkl')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
