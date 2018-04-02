#-*- coding:utf-8 -*-
from generate_face import *
from gan_model import ganModel
import tensorflow as tf

if __name__ == '__main__':
    hparams = tf.contrib.training.HParams(
        data_root = './img_align_celeba',
        crop_h = 108,    #对原始图片裁剪后高
        crop_w = 108,    #对原始图片裁剪后宽
        resize_h = 64,   #对裁剪后图片缩放的高
        resize_w = 64,   #对裁剪图片缩放的宽
        is_crop = True,  #是否裁剪
        z_dim = 100,     #随机噪声z的维度，用户generator生成图片
        batch_size = 64, #批次
        sample_size = 64,#选取作为测试样本
        output_h = 64,   #generator生成图片的高
        output_w = 64,   #generator生成图片的宽
        gf_dim = 64,     #generator的feature map的deep
        df_dim = 64)     #discriminator的feature map的deep
    face = generateFace(hparams)
    sample_images,sample_z = face.get_sample(hparams.sample_size)
    is_training = tf.placeholder(tf.bool,name='is_training')
    images = tf.placeholder(tf.float32, [None,hparams.resize_h,hparams.output_w,3],name='real_images')
    z = tf.placeholder(tf.float32, [None,hparams.z_dim], name='z')
    model = ganModel(hparams)
    g_loss,d_loss,g_vars,d_vars,g_sum,d_sum,G = model.build_model(is_training,images,z)
    d_optim,g_optim = model.optimizer(g_loss,d_loss,g_vars,d_vars)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./ckpt')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter("train_gan", sess.graph)
        step = 0
        while True:
            step = model.global_step.eval()
            batch_images,batch_z = face.next_batch(hparams.batch_size)
            #Update D network
            _, summary_str = sess.run([d_optim,d_sum],
                                           feed_dict={images:batch_images, z:batch_z, is_training:True})
            summary_writer.add_summary(summary_str,step)

            #Update G network
            _, summary_str = sess.run([g_optim,g_sum],
                                           feed_dict={z:batch_z, is_training:True})
            summary_writer.add_summary(summary_str,step)

            d_err = d_loss.eval({images:batch_images, z:batch_z, is_training:False})
            g_err = g_loss.eval({z:batch_z,is_training:False})
            print("step:%d,d_loss:%f,g_loss:%f" % (step,d_err,g_err))
            if step%1000 == 0:
                samples, d_err, g_err = sess.run([G,d_loss,g_loss],
                                                   feed_dict={images:sample_images, z:sample_z, is_training:False})
                print("sample step:%d,d_err:%f,g_err:%f" % (step,d_err,g_err))
                save_images(samples,image_manifold_size(samples.shape[0]), './samples/train_{:d}.png'.format(step))
                saver.save(sess,"./ckpt/gan.ckpt",global_step = step)