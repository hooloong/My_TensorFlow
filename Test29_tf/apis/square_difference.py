'''
tf.squared_difference

squared_difference(
    x,
    y,
    name=None
)

功能说明：
计算张量 x、y 对应元素差平方
参数列表：
参数名 	必选 	类型 	说明
x 	是 	张量 	是 half, float32, float64, int32, int64, complex64, complex128 其中一种类型
y 	是 	张量 	是 half, float32, float64, int32, int64, complex64, complex128 其中一种类型
name 	否 	string 	运算名称
'''
#!/usr/bin/python

import tensorflow as tf
import numpy as np

initial_x = [[1.,1.],[2.,2.]]
x = tf.Variable(initial_x,dtype=tf.float32)
initial_y = [[3.,3.],[4.,4.]]
y = tf.Variable(initial_y,dtype=tf.float32)
diff = tf.squared_difference(x,y)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(diff))