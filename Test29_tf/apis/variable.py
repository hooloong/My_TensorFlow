'''
tf.Variable

__init__(
    initial_value=None,
    trainable=True,
    collections=None,
    validate_shape=True,
    caching_device=None,
    name=None,
    variable_def=None,
    dtype=None,
    expected_shape=None,
    import_scope=None
)

功能说明：
维护图在执行过程中的状态信息，例如神经网络权重值的变化。
参数列表：
参数名 	类型 	说明
initial_value 	张量 	Variable 类的初始值，这个变量必须指定 shape 信息，否则后面 validate_shape 需设为 False
trainable 	Boolean 	是否把变量添加到 collection GraphKeys.TRAINABLE_VARIABLES 中（collection 是一种全局存储，不受变量名生存空间影响，一处保存，到处可取）
collections 	Graph collections 	全局存储，默认是 GraphKeys.GLOBAL_VARIABLES
validate_shape 	Boolean 	是否允许被未知维度的 initial_value 初始化
caching_device 	string 	指明哪个 device 用来缓存变量
name 	string 	变量名
dtype 	dtype 	如果被设置，初始化的值就会按照这个类型初始化
expected_shape 	TensorShape 	要是设置了，那么初始的值会是这种维度
'''

#!/usr/bin/python

import tensorflow as tf
initial = tf.truncated_normal(shape=[10,10],mean=0,stddev=1)
W=tf.Variable(initial)
list = [[1.,1.],[2.,2.]]
X = tf.Variable(list,dtype=tf.float32)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print ("##################(1)################")
    print (sess.run(W))
    print ("##################(2)################")
    print (sess.run(W[:2,:2]))
    op = W[:2,:2].assign(22.*tf.ones((2,2)))
    print ("###################(3)###############")
    print (sess.run(op))
    print ("###################(4)###############")
    print (W.eval(sess)) #computes and returns the value of this variable
    print ("####################(5)##############")
    print (W.eval())  #Usage with the default session
    print ("#####################(6)#############")
    print (W.dtype)
    print (sess.run(W.initial_value))
    print (sess.run(W.op))
    print (W.shape)
    print ("###################(7)###############")
    print (sess.run(X))