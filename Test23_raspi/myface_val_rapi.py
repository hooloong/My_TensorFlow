import tensorflow as tf #1.6
import cv2
import sys

#加了一个 intel的计算棒能帮助推理
X = tf.placeholder(tf.float32, [None, 64, 64, 3])  # 图片大小64x64 channel=3
Y = tf.placeholder(tf.float32, [None, 2])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)


def panda_joke_cnn():
    W_c1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
    b_c1 = tf.Variable(tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, W_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob_5)

    W_c2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    b_c2 = tf.Variable(tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, W_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob_5)

    W_c3 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01))
    b_c3 = tf.Variable(tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, W_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob_5)

    # Fully connected layer
    W_d = tf.Variable(tf.random_normal([8 * 16 * 32, 512], stddev=0.01))
    b_d = tf.Variable(tf.random_normal([512]))
    dense = tf.reshape(conv3, [-1, W_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, W_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob_75)

    W_out = tf.Variable(tf.random_normal([512, 2], stddev=0.01))
    b_out = tf.Variable(tf.random_normal([2]))
    out = tf.add(tf.matmul(dense, W_out), b_out)
    return out


output = panda_joke_cnn()
predict = tf.argmax(output, 1)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint('.'))


def is_my_face(image):
    res = sess.run(predict, feed_dict={X: [image / 255.0], keep_prob_5: 1.0, keep_prob_75: 1.0})
    if res[0] == 1:
        return True
    else:
        return False


face_haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_haar.detectMultiScale(gray_image, 1.3, 5)
    for face_x, face_y, face_w, face_h in faces:
        face = img[face_y:face_y + face_h, face_x:face_x + face_w]

        face = cv2.resize(face, (64, 64))

        print(is_my_face(face))

        cv2.imshow('img', face)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            sys.exit(0)

sess.close()
