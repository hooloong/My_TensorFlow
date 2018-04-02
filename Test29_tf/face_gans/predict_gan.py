#-*- coding:utf-8 -*-
from generate_face import *
from gan_model import ganModel
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    hparams = tf.contrib.training.HParams(
        z_dim = 100,
        batch_size = 1,
        gf_dim = 64,
        df_dim = 64,
        output_h = 64,
        output_w = 64)

    is_training = tf.placeholder(tf.bool,name='is_training')
    z = tf.placeholder(tf.float32, [None,hparams.z_dim], name='z')
    sample_z = np.random.uniform(-1,1,size=(hparams.batch_size,hparams.z_dim))
    model = ganModel(hparams)
    G = model.generator(z,is_training)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,"gan.ckpt-130000")
        samples = sess.run(G,feed_dict={z:sample_z,is_training:False})
        save_images(samples,image_manifold_size(samples.shape[0]),'face.png')
        print("done")