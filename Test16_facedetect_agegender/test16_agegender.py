import os
import glob
import tensorflow as tf  # 1.6
from tensorflow.contrib.layers import *
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
import numpy as np
from random import shuffle

age_table = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
sex_table = ['f', 'm']  # f:女; m:男

# AGE==True 训练年龄模型，False,训练性别模型
AGE = False

if AGE == True:
    lables_size = len(age_table)  # 年龄
else:
    lables_size = len(sex_table)  # 性别

face_set_fold = '../../data/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification'

fold_0_data = os.path.join(face_set_fold, 'fold_0_data.txt')
fold_1_data = os.path.join(face_set_fold, 'fold_1_data.txt')
fold_2_data = os.path.join(face_set_fold, 'fold_2_data.txt')
fold_3_data = os.path.join(face_set_fold, 'fold_3_data.txt')
fold_4_data = os.path.join(face_set_fold, 'fold_4_data.txt')

face_image_set = os.path.join(face_set_fold, 'aligned')


def parse_data(fold_x_data):
    data_set = []

    with open(fold_x_data, 'r',encoding="utf-8") as f:
        line_one = True
        for line in f:
            tmp = []
            if line_one == True:
                line_one = False
                continue

            tmp.append(line.split('\t')[0])
            tmp.append(line.split('\t')[1])
            tmp.append(line.split('\t')[3])
            tmp.append(line.split('\t')[4])

            file_path = os.path.join(face_image_set, tmp[0])
            if os.path.exists(file_path):
                filenames = glob.glob(file_path + "/*.jpg")
                for filename in filenames:
                    if tmp[1] in filename:
                        break
                if AGE == True:
                    if tmp[2] in age_table:
                        data_set.append([filename, age_table.index(tmp[2])])
                else:
                    if tmp[3] in sex_table:
                        data_set.append([filename, sex_table.index(tmp[3])])

    return data_set


data_set_0 = parse_data(fold_0_data)
data_set_1 = parse_data(fold_1_data)
data_set_2 = parse_data(fold_2_data)
data_set_3 = parse_data(fold_3_data)
data_set_4 = parse_data(fold_4_data)

data_set = data_set_0 + data_set_1 + data_set_2 + data_set_3 + data_set_4
shuffle(data_set)

# 缩放图像的大小
IMAGE_HEIGHT = 227
IMAGE_WIDTH = 227
# 读取缩放图像
jpg_data = tf.placeholder(dtype=tf.string)
decode_jpg = tf.image.decode_jpeg(jpg_data, channels=3)
resize = tf.image.resize_images(decode_jpg, [IMAGE_HEIGHT, IMAGE_WIDTH])
resize = tf.cast(resize, tf.uint8) / 255


def resize_image(file_name):
    with tf.gfile.FastGFile(file_name, 'rb') as f:
        image_data = f.read()
    with tf.Session() as sess:
        image = sess.run(resize, feed_dict={jpg_data: image_data})
    return image


pointer = 0


# 应该先处理好图片或使用string_input_producer
def get_next_batch(data_set, batch_size=128):
    global pointer
    batch_x = []
    batch_y = []
    for i in range(batch_size):
        batch_x.append(resize_image(data_set[pointer][0]))
        batch_y.append(data_set[pointer][1])
        pointer += 1
    return batch_x, batch_y


batch_size = 128
num_batch = len(data_set) // batch_size

X = tf.placeholder(dtype=tf.float32, shape=[batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
Y = tf.placeholder(dtype=tf.int32, shape=[batch_size])


def conv_net(nlabels, images, pkeep=1.0):
    weights_regularizer = tf.contrib.layers.l2_regularizer(0.0005)
    with tf.variable_scope("conv_net", "conv_net", [images]) as scope:
        with tf.contrib.slim.arg_scope([convolution2d, fully_connected], weights_regularizer=weights_regularizer,
                                       biases_initializer=tf.constant_initializer(1.),
                                       weights_initializer=tf.random_normal_initializer(stddev=0.005), trainable=True):
            with tf.contrib.slim.arg_scope([convolution2d],
                                           weights_initializer=tf.random_normal_initializer(stddev=0.01)):
                conv1 = convolution2d(images, 96, [7, 7], [4, 4], padding='VALID',
                                      biases_initializer=tf.constant_initializer(0.), scope='conv1')
                pool1 = max_pool2d(conv1, 3, 2, padding='VALID', scope='pool1')
                norm1 = tf.nn.local_response_normalization(pool1, 5, alpha=0.0001, beta=0.75, name='norm1')
                conv2 = convolution2d(norm1, 256, [5, 5], [1, 1], padding='SAME', scope='conv2')
                pool2 = max_pool2d(conv2, 3, 2, padding='VALID', scope='pool2')
                norm2 = tf.nn.local_response_normalization(pool2, 5, alpha=0.0001, beta=0.75, name='norm2')
                conv3 = convolution2d(norm2, 384, [3, 3], [1, 1], biases_initializer=tf.constant_initializer(0.),
                                      padding='SAME', scope='conv3')
                pool3 = max_pool2d(conv3, 3, 2, padding='VALID', scope='pool3')
                flat = tf.reshape(pool3, [-1, 384 * 6 * 6], name='reshape')
                full1 = fully_connected(flat, 512, scope='full1')
                drop1 = tf.nn.dropout(full1, pkeep, name='drop1')
                full2 = fully_connected(drop1, 512, scope='full2')
                drop2 = tf.nn.dropout(full2, pkeep, name='drop2')
    with tf.variable_scope('output') as scope:
        weights = tf.Variable(tf.random_normal([512, nlabels], mean=0.0, stddev=0.01), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[nlabels], dtype=tf.float32), name='biases')
        output = tf.add(tf.matmul(drop2, weights), biases, name=scope.name)
    return output


def training():
    logits = conv_net(lables_size, X)

    def optimizer(eta, loss_fn):
        global_step = tf.Variable(0, trainable=False)
        optz = lambda lr: tf.train.MomentumOptimizer(lr, 0.9)
        lr_decay_fn = lambda lr, global_step: tf.train.exponential_decay(lr, global_step, 100, 0.97, staircase=True)
        return tf.contrib.layers.optimize_loss(loss_fn, global_step, eta, optz, clip_gradients=4.,
                                               learning_rate_decay_fn=lr_decay_fn)

    def loss(logits, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = cross_entropy_mean + 0.01 * sum(regularization_losses)
        loss_averages = tf.train.ExponentialMovingAverage(0.9)
        loss_averages_op = loss_averages.apply([cross_entropy_mean] + [total_loss])
        with tf.control_dependencies([loss_averages_op]):
            total_loss = tf.identity(total_loss)
        return total_loss

    # loss
    total_loss = loss(logits, Y)
    # optimizer
    train_op = optimizer(0.001, total_loss)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        global pointer
        epoch = 0
        while True:
            pointer = 0
            for batch in range(num_batch):
                batch_x, batch_y = get_next_batch(data_set, batch_size)
                _, loss_value = sess.run([train_op, total_loss], feed_dict={X: batch_x, Y: batch_y})
                print(epoch, batch, loss_value)
            saver.save(sess, 'age.module' if AGE == True else 'sex.module')
            epoch += 1


training()

"""
# 检测性别和年龄
# 把batch_size改为1
def detect_age_or_sex(image_path):
	logits = conv_net(lables_size, X)
	saver = tf.train.Saver()

	with tf.Session() as sess:
		saver.restore(sess, './age.module' if AGE == True else './sex.module')

		softmax_output = tf.nn.softmax(logits)
		res = sess.run(softmax_output, feed_dict={X:[resize_image(image_path)]})
		res = np.argmax(res)

		if AGE == True:
			return age_table[res]
		else:
			return sex_table[res]
"""