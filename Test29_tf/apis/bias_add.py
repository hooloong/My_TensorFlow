'''
tf.nn.bias_add

bias_add(
    value,
    bias,
    data_format=None,
    name=None
)

功能说明：
将偏差项 bias 加到 value 上面，可以看做是 tf.add 的一个特例，其中 bias 必须是一维的，并且维度和 value 的最后一维相同，数据类型必须和 value 相同。
参数列表：
参数名 	必选 	类型 	说明
value 	是 	张量 	数据类型为 float, double, int64, int32, uint8, int16, int8, complex64, or complex128
bias 	是 	1 维张量 	维度必须和 value 最后一维维度相等
data_format 	否 	string 	数据格式，支持 ' NHWC ' 和 ' NCHW '
name 	否 	string 	运算名称
'''
#!/usr/bin/python

import tensorflow as tf
import numpy as np

a = tf.constant([[1.0, 2.0],[1.0, 2.0],[1.0, 2.0]])
b = tf.constant([2.0,1.0])
c = tf.constant([1.0])
sess = tf.Session()
print (sess.run(tf.nn.bias_add(a, b)))
#print (sess.run(tf.nn.bias_add(a,c))) error
print ("##################################")
print (sess.run(tf.add(a, b)))
print ("##################################")
print (sess.run(tf.add(a, c)))