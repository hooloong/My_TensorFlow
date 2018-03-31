'''
relu(
    features,
    name=None
)

功能说明：
relu激活函数可以参考 CS231n: Convolutional Neural Networks for Visual Recognition
参数列表：
参数名 	必选 	类型 	说明
features 	是 	tensor 	是以下类型float32, float64, int32, int64, uint8, int16, int8, uint16, half
name 	否 	string 	运算名称
'''
import tensorflow as tf

a = tf.constant([1,-2,0,4,-5,6])
b = tf.nn.relu(a)
with tf.Session() as sess:
    print (sess.run(b))