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
from PIL import Image, ImageFilter

def load_data(argv):

    grayimage = Image.open(argv).convert('L')
    width = float(grayimage.size[0])
    height = float(grayimage.size[1])
    newImage = Image.new('L', (28, 28), (255))

    if width > height:
        nheight = int(round((20.0/width*height),0))
        if (nheigth == 0):
            nheigth = 1
        img = grayimage.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0))
        newImage.paste(img, (4, wtop))
    else:
        nwidth = int(round((20.0/height*width),0))
        if (nwidth == 0):
            nwidth = 1
        img = grayimage.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0))
        newImage.paste(img, (wleft, 4))

    tv = list(newImage.getdata())
    tva = [ (255-x)*1.0/255.0 for x in tv]
    return tva

def main(argv):

    imvalue = load_data(argv)

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    y_conv, keep_prob = mnist_model.deepnn(x)

    y_predict = tf.nn.softmax(y_conv)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "mnist_cnn_model.ckpt")
        prediction=tf.argmax(y_predict,1)
        predint = prediction.eval(feed_dict={x: [imvalue],keep_prob: 1.0}, session=sess)
        print (predint[0])

if __name__ == "__main__":
    main(sys.argv[1])