#-*- coding:utf-8 -*-
import tensorflow as tf
import math

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name
    def __call__(self,x,train):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon,
                                            center=True, scale=True, is_training=train, scope=self.name)
class ganModel:
    def __init__(self,hparams):
        self.batch_size = hparams.batch_size
        self.gf_dim = hparams.gf_dim
        self.df_dim = hparams.df_dim
        self.output_h = hparams.output_h
        self.output_w = hparams.output_w
        #batch normalization
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.global_step = tf.Variable(1, trainable=False)

    def linear(self,input_z,output_size,scope=None, stddev=0.02, bias_start=0.0):
        shape = input_z.get_shape().as_list()
        with tf.variable_scope(scope or "linear"):
            matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev))
            bias = tf.get_variable("bias", [output_size],
                                   initializer=tf.constant_initializer(bias_start))
            return tf.matmul(input_z,matrix) + bias

    def conv2d_transpose(self,input_, output_shape,
                         k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
                         name="conv2d_transpose"):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])
            biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
            deconv = tf.nn.bias_add(deconv, biases)
            return deconv

    def conv2d(self,image,output_dim,
               k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
               name="conv2d"):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, image.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(image, w, strides=[1, d_h, d_w, 1], padding='SAME')
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
            return conv

    def lrelu(self,x, leak=0.2, name="lrelu"):
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)

    def conv_out_size_same(self,size, stride):
        return int(math.ceil(float(size) / float(stride)))

    def generator(self,z,is_training):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.output_h, self.output_w  #64*64
            s_h2, s_w2 = self.conv_out_size_same(s_h, 2), self.conv_out_size_same(s_w, 2) #32*32
            s_h4, s_w4 = self.conv_out_size_same(s_h2, 2), self.conv_out_size_same(s_w2, 2) #16*16
            s_h8, s_w8 = self.conv_out_size_same(s_h4, 2), self.conv_out_size_same(s_w4, 2) #8*8
            s_h16, s_w16 = self.conv_out_size_same(s_h8, 2), self.conv_out_size_same(s_w8, 2) #4*4

            z_ = self.linear(z,self.gf_dim*8*s_h16*s_w16,'g_h0_lin')
            h0 = tf.reshape(z_,[-1,s_h16,s_w16,self.gf_dim*8])
            h0 = tf.nn.relu(self.g_bn0(h0,is_training))

            h1 = self.conv2d_transpose(h0,[self.batch_size,s_h8,s_w8,self.gf_dim*4],name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1,is_training))

            h2 = self.conv2d_transpose(h1,[self.batch_size,s_h4,s_w4,self.gf_dim*2],name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2,is_training))

            h3 = self.conv2d_transpose(h2,[self.batch_size,s_h2,s_w2,self.gf_dim*1],name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3,is_training))

            h4 = self.conv2d_transpose(h3,[self.batch_size,s_h,s_w,3],name='g_h4')

            return tf.nn.tanh(h4)

    def discriminator(self,image,is_training,reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = self.lrelu(self.conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = self.lrelu(self.d_bn1(self.conv2d(h0, self.df_dim*2, name='d_h1_conv'), is_training))
            h2 = self.lrelu(self.d_bn2(self.conv2d(h1, self.df_dim*4, name='d_h2_conv'), is_training))
            h3 = self.lrelu(self.d_bn3(self.conv2d(h2, self.df_dim*8, name='d_h3_conv'), is_training))
            h3 = tf.reshape(h3,[-1,8192]) #8192 = self.df_dim*8*4*4
            h4 = self.linear(h3,1,'d_h4_lin')

            return tf.nn.sigmoid(h4), h4

    def build_model(self,is_training,images,z):
        z_sum = tf.summary.histogram("z",z)

        G = self.generator(z,is_training)
        D,D_logits = self.discriminator(images,is_training)
        D_,D_logits_ = self.discriminator(G,is_training,reuse=True)

        d_sum = tf.summary.histogram("d",D)
        d__sum = tf.summary.histogram("d_",D_)
        G_sum = tf.summary.image("G", G)

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits,
                                                    labels=tf.ones_like(D)))#对于discriminator，尽量判断images是货真价实
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_,
                                                    labels=tf.zeros_like(D_)))#对于discriminator，尽量判断G是伪冒

        d_loss_real_sum = tf.summary.scalar("d_loss_real",d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake",d_loss_fake)

        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_,
                                                    labels=tf.ones_like(D_)))#对于generator，尽量然D判断G是货真价实的
        d_loss = d_loss_real + d_loss_fake

        g_loss_sum = tf.summary.scalar("g_loss", g_loss)
        d_loss_sum = tf.summary.scalar("d_loss", d_loss)

        t_vars = tf.trainable_variables() #discriminator、generator两个网络参数分开训练
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        g_sum = tf.summary.merge([z_sum,d__sum,G_sum,d_loss_fake_sum,g_loss_sum])
        d_sum = tf.summary.merge([z_sum,d_sum,d_loss_real_sum,d_loss_sum])

        return g_loss,d_loss,g_vars,d_vars,g_sum,d_sum,G

    def optimizer(self,g_loss,d_loss,g_vars,d_vars,learning_rate = 0.0002,beta1=0.5):
        d_optim = tf.train.AdamOptimizer(learning_rate,beta1=beta1).minimize(d_loss,global_step=self.global_step,var_list=d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate,beta1=beta1).minimize(g_loss,var_list=g_vars)
        return d_optim,g_optim