#!/usr/bin/python
# -*- coding: utf-8 -*

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import mnist_model

FLAGS = None


def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  #输入变量，mnist图片大小为28*28
  x = tf.placeholder(tf.float32, [None, 784])

  #输出变量，数字是1-10
  y_ = tf.placeholder(tf.float32, [None, 10])

  # 构建网络，输入—>第一层卷积—>第一层池化—>第二层卷积—>第二层池化—>第一层全连接—>第二层全连接
  y_conv, keep_prob = mnist_model.deepnn(x)

  #第一步对网络最后一层的输出做一个softmax，第二步将softmax输出和实际样本做一个交叉熵
  #cross_entropy返回的是向量
  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)

  #求cross_entropy向量的平均值得到交叉熵
  cross_entropy = tf.reduce_mean(cross_entropy)

  #AdamOptimizer是Adam优化算法：一个寻找全局最优点的优化算法，引入二次方梯度校验
  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  #在测试集上的精确度
  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  #将神经网络图模型保存本地，可以通过浏览器查看可视化网络结构
  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  #将训练的网络保存下来
  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})#输入是字典，表示tensorflow被feed的值
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    test_accuracy = 0
    for i in range(200):
      batch = mnist.test.next_batch(50)
      test_accuracy += accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}) / 200;

    print('test accuracy %g' % test_accuracy)

    save_path = saver.save(sess,"./mnist_cnn_model.ckpt")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='../../../data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)