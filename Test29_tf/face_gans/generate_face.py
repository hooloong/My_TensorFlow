#-*- coding:utf-8 -*-
import itertools
import os
from glob import glob
import numpy as np
import scipy.misc
import tensorflow as tf


def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def imsave(images, size, path):
    img = merge(images, size)
    return scipy.misc.imsave(path, (255*img).astype(np.uint8))

def inverse_transform(images):
    return (images+1.)/2.

def save_images(images,size,image_path):
    return imsave(inverse_transform(images), size, image_path)

class generateFace:
    def __init__(self,hparams):
        self.formats = ["png","jpg","jpeg"]
        self.datas_path = self.get_datas_path(hparams.data_root)
        self.datas_size = len(self.datas_path)
        self.crop_h = hparams.crop_h
        self.crop_w = hparams.crop_w
        self.resize_h = hparams.resize_h
        self.resize_w = hparams.resize_w
        self.is_crop = hparams.is_crop
        self.z_dim = hparams.z_dim
        self._index_in_epoch = 0

    def get_datas_path(self,data_root):
        return list(itertools.chain.from_iterable(
            glob(os.path.join(data_root,"*.{}".format(ext))) for ext in self.formats))

    def get_image(self,path):
        img = scipy.misc.imread(path,mode='RGB').astype(np.float)
        if(self.is_crop): #截取中间部分
            h,w = img.shape[:2] #图像宽、高
            assert(h > self.crop_h and w > self.crop_w)
            j = int(round((h - self.crop_h)/2.))
            i = int(round((w - self.crop_w)/2.))
            img = img[j:j+self.crop_h,i:i+self.crop_w]
        img = scipy.misc.imresize(img,[self.resize_h,self.resize_w])
        return np.array(img)/127.5 - 1.

    def get_batch(self,batch_files):
        batch_images = [self.get_image(path) for path in batch_files]
        batch_images = np.array(batch_images).astype(np.float32)
        batch_z = np.random.uniform(-1,1,size=(len(batch_files),self.z_dim))
        return batch_images,batch_z

    def get_sample(self,sample_size):
        assert(self.datas_size > sample_size)
        np.random.shuffle(self.datas_path)
        sample_files = self.datas_path[0:sample_size]
        return self.get_batch(sample_files)

    def next_batch(self,batch_size):
        assert(self.datas_size > batch_size)
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if(self._index_in_epoch > self.datas_size):
            np.random.shuffle(self.datas_path)
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        batch_files = self.datas_path[start:end]
        return self.get_batch(batch_files)

'''
hparams = tf.contrib.training.HParams(
    data_root = './img_align_celeba',
    crop_h = 108,
    crop_w = 108,
    resize_h = 64,
    resize_w = 64,
    is_crop = True,
    z_dim = 100,
    batch_size = 64,
    sample_size = 64,
    output_h = 64,
    output_w = 64,
    gf_dim = 64,
    df_dim = 64)
face = generateFace(hparams)

img,z = face.next_batch(1)
z
save_images(img,(1,1),"test.jpg")
'''