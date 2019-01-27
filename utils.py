from six import iteritems
import os
import tensorflow as tf
from preprocessing import inception_preprocessing
import csv
import pickle


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))


def get_image_paths(facedir):
    """ 获得一个目录中所有的图像路径 """
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


def get_dataset(path, has_class_directories=True):
    """ 按类别从文件夹中获得分类对象,返回ImageClass的list """
    imgset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp)
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()

    nrof_classes = len(classes)

    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        imgset.append(ImageClass(class_name, image_paths))

    return imgset


def get_image_paths_and_labels(imgset):
    # 从一个imgset里面获得所有的imagepath以及label
    image_paths_flat = []
    labels_flat = []
    for i in range(len(imgset)):
        image_paths_flat += imgset[i].image_paths
        labels_flat += [i] * len(imgset[i].image_paths)
    return image_paths_flat, labels_flat


def parser(filename, label, class_num, height, witdh, is_training):
    # with tf.gfile.GFile(filename, 'rb') as f:
    img = tf.read_file(filename)  # f.read()
    img = tf.image.decode_jpeg(img, channels=3)
    # NOTE the inception_preprocessing will convert image scale to [-1,1]

    img_resized = inception_preprocessing.preprocess_image(
        img, height, witdh, is_training=is_training,
        add_image_summaries=False)

    one_hot_label = tf.one_hot(label, class_num, 1, 0)
    # NOTE label should expand axis
    # one_hot_label = one_hot_label[tf.newaxis, tf.newaxis, :]
    return img_resized, one_hot_label


def create_dataset(namelist: list, labelist: list, batchsize: int, class_num: int,
                   height: int, witdh: int, seed: int, is_training=True):
    """ 构建Tensorflow dataset"""
    # create the dataset from the list
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(namelist), tf.constant(labelist)))
    # parser the data set
    dataset = dataset.apply(tf.data.experimental.map_and_batch(
        map_func=lambda filename, label:
        parser(filename, label, class_num,
               height, witdh, is_training),
        batch_size=batchsize,
        # add drop_remainder avoid output shape less than batchsize
        drop_remainder=True))
    # repeat
    dataset = dataset.repeat()
    # shuffle
    dataset = dataset.shuffle(50, seed=seed)
    # clac step for per epoch
    step_for_epoch = int(len(labelist)/batchsize)
    return dataset, step_for_epoch


def create_dataset_iter(dataset: tf.data.Dataset):
    """ create dataset iter

    Parameters
    ----------
    dataset : tf.data.Dataset

    Returns
    -------
    dataset iter
    """
    data_it = dataset.make_one_shot_iterator()
    # 定义个获取下一组数据的操作(operator)
    return data_it.get_next()


def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                if par[1] == '-':
                    lr = -1
                else:
                    lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate


def get_control_flag(control, field):
    return tf.equal(tf.mod(tf.floor_div(control, field), 2), 1)


def _add_loss_summaries(total_loss):
    """Add summaries for losses.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply([total_loss])

    return loss_averages_op


def create_train_op(total_loss, global_step, optimizer, learning_rate, moving_average_decay):
    # Generate moving averages of all losses and associated summaries.
    # loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    # with tf.control_dependencies([loss_averages_op]):
    if optimizer == 'ADAGRAD':
        opt = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer == 'ADADELTA':
        opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
    elif optimizer == 'ADAM':
        opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
    elif optimizer == 'RMSPROP':
        opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
    elif optimizer == 'MOM':
        opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True)
    else:
        raise ValueError('Invalid optimization algorithm')
    # ! 因为minimize中包含了compute_gradients和apply_gradients,所以他们的思路是:
    # ! 首先计算梯度,然后给梯度增加滑动平均,最后把滑动平均的梯度应用到梯度下降
    #     grads = opt.compute_gradients(total_loss, update_gradient_vars)

    # # Apply gradients.
    # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # # Track the moving averages of all trainable variables.
    # variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    # variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    #     train_op = tf.no_op(name='train')
    # ! 我先用原始的方式进行训练
    train_op = opt.minimize(total_loss, global_step)

    return train_op


def load_pkl(filepath)->dict:
    """ 加载pkl文件 """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pkl(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def restore_form_pkl(sess: tf.Session(), pklpath: str, except_last=True):
    """ restore the pre-train weight form the .pkl file

    Parameters
    ----------
    sess : tf.Session
        sess
    pklpath : str
        .pkl file path
    except_last : bool, optional
        whether load the last layer weight, when you custom the net shouldn't load 
        the layer name scope is 'MobileNetV1/Bottleneck2'
        (the default is True, which not load the last layer weight)
    """
    # tf.global_variables() == tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # filter the last layer weight
    modelvarlist = [var for var in tf.trainable_variables(scope='MobileNetV1') if not (except_last and 'MobileNetV1/Bottleneck2' in var.name)]
    pre_weight_dict = load_pkl(pklpath)

    # make sure the number equal
    var_num = len(modelvarlist)

    # save the opt to list
    opt_list = []
    for newv in modelvarlist:
        for k, oldv in pre_weight_dict.items():
            if k == newv.name:
                opt_list.append(tf.assign(newv, oldv))

    # make sure the number equal
    assert len(opt_list) == var_num
    # run the assign
    sess.run(opt_list)
