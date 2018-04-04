import tensorflow as tf
import numpy as np
import glob
import os
from PIL import Image


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def encode_to_tfrecords(data_path, name, rows=24, cols=16):  # 从图片路径读取图片编码成tfrecord
    folders = os.listdir(data_path)  # 这里都和我图片的位置有关
    folders.sort()
    numclass = len(folders)
    i = 0
    npic = 0
    writer = tf.python_io.TFRecordWriter(name)
    for floder in folders:
        path = data_path + "/" + floder
        img_names = glob.glob(os.path.join(path, "*.bmp"))
        for img_name in img_names:
            img_path = img_name
            img = Image.open(img_path).convert('P')
            img = img.resize((cols, rows))
            img_raw = img.tobytes()
            labels = [0] * 34  # 我用的是softmax，要和预测值的维度一致
            labels[i] = 1
            example = tf.train.Example(features=tf.train.Features(feature={  # 填充example
                'image_raw': _byte_feature(img_raw),
                'label': _int64_feature(labels)}))
            writer.write(example.SerializeToString())  # 把example加入到writer里，最后写到磁盘。
            npic = npic + 1
        i = i + 1
    writer.close()
    print (npic)


def decode_from_tfrecord(filequeuelist, rows=24, cols=16):
    reader = tf.TFRecordReader()  # 文件读取
    _, example = reader.read(filequeuelist)
    features = tf.parse_single_example(example, features={'image_raw':  # 解码
                                                              tf.FixedLenFeature([], tf.string),
                                                          'label': tf.FixedLenFeature([34, 1], tf.int64)})
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape(rows * cols)
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return image, label


def get_batch(filename_queue, batch_size):
    with tf.name_scope('get_batch'):
        [image, label] = decode_from_tfrecord(filename_queue)
        images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=2,
                                                capacity=100 + 3 * batch_size, min_after_dequeue=100)
        return images, labels


def generate_filenamequeue(filequeuelist):
    filename_queue = tf.train.string_input_producer(filequeuelist, num_epochs=5)
    return filename_queue


def test(filename, batch_size):
    filename_queue = generate_filenamequeue(filename)
    [images, labels] = get_batch(filename_queue, batch_size)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess = tf.InteractiveSession()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    try:
        while not coord.should_stop():
            image, label = sess.run([images, labels])
            i = i + 1
            if i % 1000 == 0:
                for j in range(batch_size):  # 之前tfrecord编码的时候，数据范围变成[-0.5,0.5],现在相当于逆操作，把数据变成图片像素值
                    image[j] = (image[j] + 0.5) * 255
                    ar = np.asarray(image[j], np.uint8)
                    # image[j]=tf.cast(image[j],tf.uint8)
                    print
                    ar.shape
                    img = Image.frombytes("P", (16, 24), ar.tostring())  # 函数参数中宽度高度要注意。构建24×16的图片
                    img.save("../../../data/MNIST_data/reverse_%d.bmp" % (i + j), "BMP")  # 保存部分图片查看
            '''if(i>710):
                print("step %d"%(i))
                print image
                print label'''
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()