'''
tf.reduce_mean

reduce_mean(
    input_tensor,
    axis=None,
    keep_dims=False,
    name=None,
    reduction_indices=None
)

功能说明：
计算张量 input_tensor 平均值
参数列表：
参数名 	必选 	类型 	说明
input_tensor 	是 	张量 	输入待求平均值的张量
axis 	否 	None、0、1 	None：全局求平均值；0：求每一列平均值；1：求每一行平均值
keep_dims 	否 	Boolean 	保留原来的维度(例如不会从二维矩阵降为一维向量)
name 	否 	string 	运算名称
reduction_indices 	否 	None 	和 axis 等价，被弃用
'''

#!/usr/bin/python

import tensorflow as tf
import numpy as np

initial = [[1.,1.],[2.,2.]]
x = tf.Variable(initial,dtype=tf.float32)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(tf.reduce_mean(x)))
    print(sess.run(tf.reduce_mean(x,0))) #Column
    print(sess.run(tf.reduce_mean(x,1))) #row