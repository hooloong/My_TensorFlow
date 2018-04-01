#-*- coding:utf-8 -*-
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

MNIST = input_data.read_data_sets("../../../data", one_hot=True)

learning_rate = 0.01
batch_size = 128
n_epochs = 25

X = tf.placeholder(tf.float32, [batch_size, 784])
Y = tf.placeholder(tf.float32, [batch_size, 10])

w = tf.Variable(tf.random_normal(shape=[784,10], stddev=0.01), name="weights")
b = tf.Variable(tf.zeros([1, 10]), name="bias")

logits = tf.matmul(X, w) + b

entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
loss = tf.reduce_mean(entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    n_batches = int(MNIST.train.num_examples/batch_size)
    for i in range(n_epochs):
        for j in range(n_batches):
            X_batch, Y_batch = MNIST.train.next_batch(batch_size)
            _, loss_ = sess.run([optimizer, loss], feed_dict={ X: X_batch, Y: Y_batch})
            print("Loss of epochs[{0}] batch[{1}]: {2}".format(i, j, loss_))

    n_batches = int(MNIST.test.num_examples/batch_size)
    total_correct_preds = 0
    for i in range(n_batches):
        X_batch, Y_batch = MNIST.test.next_batch(batch_size)
        preds = tf.nn.softmax(tf.matmul(X_batch, w) + b)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

        total_correct_preds += sess.run(accuracy)

    print("Accuracy {0}".format(total_correct_preds/MNIST.test.num_examples))