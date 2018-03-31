'''
sigmoid_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    name=None
)

功能说明：
先对 logits 通过 sigmoid 计算，再计算交叉熵，交叉熵代价函数可以参考 CS231n: Convolutional Neural Networks for Visual Recognition
参数列表：
参数名 	必选 	类型 	说明
_sentinel 	否 	None 	没有使用的参数
labels 	否 	Tensor 	type, shape 与 logits相同
logits 	否 	Tensor 	type 是 float32 或者 float64
name 	否 	string 	运算名称
'''

import tensorflow as tf
x = tf.constant([1,2,3,4,5,6,7],dtype=tf.float64)
y = tf.constant([1,1,1,0,0,1,0],dtype=tf.float64)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = y,logits = x)
with tf.Session() as sess:
    print (sess.run(loss))