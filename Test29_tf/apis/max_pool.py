'''
max_pool(
    value,
    ksize,
    strides,
    padding,
    data_format='NHWC',
    name=None
)

功能说明：
池化的原理可参考 CS231n: Convolutional Neural Networks for Visual Recognition
参数列表：
参数名 	必选 	类型 	说明
value 	是 	tensor 	4 维的张量，即 [ batch, height, width, channels ]，数据类型为 tf.float32
ksize 	是 	列表 	池化窗口的大小，长度为 4 的 list，一般是 [1, height, width, 1]，因为不在 batch 和 channels 上做池化，所以第一个和最后一个维度为 1
strides 	是 	列表 	池化窗口在每一个维度上的步长，一般 strides[0] = strides[3] = 1
padding 	是 	string 	只能为 " VALID "，" SAME " 中之一，这个值决定了不同的池化方式。VALID 丢弃方式；SAME：补全方式
data_format 	否 	string 	只能是 " NHWC ", " NCHW "，默认" NHWC "
name 	否 	string 	运算名称
'''
import tensorflow as tf

a = tf.constant([1,3,2,1,2,9,1,1,1,3,2,3,5,6,1,2],dtype=tf.float32,shape=[1,4,4,1])
b = tf.nn.max_pool(a,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
c = tf.nn.max_pool(a,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
with tf.Session() as sess:
    print ("b shape:")
    print (b.shape)
    print ("b value:")
    print (sess.run(b))
    print ("c shape:")
    print (c.shape)
    print ("c value:")
    print (sess.run(c))