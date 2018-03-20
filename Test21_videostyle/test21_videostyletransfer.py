# -*- coding: utf-8 -*-
#
import tensorflow as tf  #1.6
import numpy as np
import cv2
import sys
import os
import argparse
import shutil
# https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip
#https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
inception_model = '../../models/tensorflow_inception_graph.pb'

# 加载模型
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

X = tf.placeholder(np.float32, name='input')
with tf.gfile.FastGFile(inception_model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
imagenet_mean = 117.0
preprocessed = tf.expand_dims(X - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': preprocessed})

layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]
feature_nums = [int(graph.get_tensor_by_name(name + ':0').get_shape()[-1]) for name in layers]

print('layers:', len(layers))  # 59
print('feature:', sum(feature_nums))  # 7548


def tffunc(*argtypes):
    placeholders = list(map(tf.placeholder, argtypes))

    def wrap(f):
        out = f(*placeholders)

        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))

        return wrapper

    return wrap


def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0, :, :, :]


# 选择层
layer = 'mixed4c'
resize = tffunc(np.float32, np.int32)(resize)
score = tf.reduce_mean(tf.square(graph.get_tensor_by_name("import/%s:0" % layer)))
gradi = tf.gradients(score, X)[0]


# Deep Dream
def deep_dream(img_noise, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    img = img_noise
    octaves = []

    for _ in range(octave_n - 1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw) / octave_scale))
        hi = img - resize(lo, hw)
        img = lo
        octaves.append(hi)

    # tile
    def calc_grad_tiled(img, t_grad, tile_size=512):
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h - sz // 2, sz), sz):
            for x in range(0, max(w - sz // 2, sz), sz):
                sub = img_shift[y:y + sz, x:x + sz]
                g = sess.run(t_grad, {X: sub})
                grad[y:y + sz, x:x + sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)

    res = None
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2]) + hi
        for _ in range(iter_n):
            g = calc_grad_tiled(img, gradi)
            img += g * (step / (np.abs(g).mean() + 1e-7))

        res = img
    return res


#####################################
# cmd : python test21.py -i ai1.mp4 -o aio.mp4
parser = argparse.ArgumentParser(description='Deep Video Videos.')
parser.add_argument('-i', '--input', help='inupt mp4 Video File Path', required=True)
parser.add_argument('-o', '--output', help='output mp4 Video File Path', required=True)

args = parser.parse_args()
print(args)

if not os.path.exists(args.input):
    print("please input video")
    sys.exit(0)


# 我使用ffmpeg把视频转为帧
def video_to_frames(video_path, frames_path):
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)
        output_file = frames_path + "/%08d.jpg"
        print("ffmpeg -i {} -f image2 {}".format(video_path, output_file))
        os.system("ffmpeg -i {} -f image2 {}".format(video_path, output_file))


# 临时目录
tmp_path = './tmp'

if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)
    video_to_frames(args.input, tmp_path + '/frames_input')
    # deep dream每一帧
    frames = [name for name in os.listdir(tmp_path + '/frames_input') if
              os.path.isfile(os.path.join(tmp_path + '/frames_input', name))]
    frames.sort()
    print("要处理的帧数:", len(frames))
    for frame in frames:
        print('正在转换: ', frame)
        img_frame = cv2.imread(tmp_path + '/frames_input/' + frame)
        img = deep_dream(img_noise=img_frame)
        if not os.path.exists(tmp_path + '/frames_output'):
            os.makedirs(tmp_path + '/frames_output')
        cv2.imwrite(tmp_path + '/frames_output/' + frame, img)
else:
    print("TODO: 从前一次退出的地方继续执行")

# TODO: 使用ffmpeg把帧转回视频
"""
# 帧率
ffprobe -show_streams -select_streams v -i args.input 2>/dev/null | grep "r_frame_rate" | cut -d'=' -f2
ffmpeg -framerate [FPS] -i ./tmp/frames_output/%08d.jpg -c:v libx264 -vf "fps=[FPS],format=yuv420p" -tune fastdecode -tune zerolatency -profile:v baseline ./tmp/tmp.mp4 -y
ffmpeg -i args.input -strict -2 ./tmp/tmp.aac -y
ffmpeg -i ./tmp/tmp.aac -i ./tmp/tmp.mp4 -strict -2 -c:v copy -movflags faststart -shortest args.output
"""

# shutil.rmtree(tmp_path)