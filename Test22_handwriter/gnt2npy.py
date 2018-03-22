# -*- coding: utf-8 -*-

from __future__ import print_function

import cv2
import numpy as np
import os
import struct
import sys
import zipfile
#http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html
def gnt2npy(src_file, dst_file, image_size, map_file):
    '''
    将gnt文件存为npy格式

    param src_file: 源文件名，gnt文件
    param dst_file: 目标文件名， 若此参数设置为'xxx'，则会生成xxx_images.npy 和 xxx_labels.npy
    param image_size: 图片大小，设置为m时，最终文件的大小将为 m x m
    param map_file: 由于汉字编码不连续，作为分类label并不合适，该文件保存汉字码和label的映射关系
    '''

    code_map = {}
    if os.path.exists(map_file):
        with open(map_file, 'r') as fp:
            for line in fp.readlines():
                if len(line) == 0:
                    continue;
                code, label = line.split()
                code_map[int(code)] = int(label)
        fp.close()
    images = []
    labels = []

    if zipfile.is_zipfile(src_file):    #单体zip文件
        zip_file = zipfile.ZipFile(src_file, 'r')
        file_list = zip_file.namelist()
        for file_name in file_list:
            print("processing %s ..." % file_name)
            data_file = zip_file.open(file_name)
            total_bytes = zip_file.getinfo(file_name).file_size
            image_list, label_list, code_map = readFromGnt(data_file,  image_size, total_bytes, code_map)
            images += image_list
            labels += label_list
    elif os.path.isdir(src_file): #包含gnt文件的文件夹
        file_list = os.listdir(src_file)
        for file_name in file_list:
            file_name = src_file + os.sep + file_name
            print("processing %s ..." % file_name)
            data_file = open(file_name, 'rb')
            total_bytes = os.path.getsize(file_name)
            image_list, label_list, code_map = readFromGnt(data_file, image_size, total_bytes, code_map)
            images += image_list
            labels += label_list
    else:
        sys.stderr.write('Source file should be a ziped file containing the gnt files. Plese check your input again.\n')
        return None

    with open(map_file, 'w') as fp:
        for code in code_map:
            print(code, code_map[code], file=fp)
    fp.close()
    np.save(dst_file + '_images.npy', images)
    np.save(dst_file + '_labels.npy', labels)


def readFromGnt(data_file, image_size, total_bytes, code_map):
    '''
    从文件对象中读取数据并返回

    param data_file， 文件对象
    param image_size:  图片大小，设置为m时，最终文件的大小将为m x m
    param total_bytes: 文件总byte数
    param code_map： 由于汉字编码不连续，作为分类label并不合适，该dict保存汉字码和label的映射关系
    '''
    decoded_bytes = 0
    image_list = []
    label_list = []
    new_label = len(code_map)
    while decoded_bytes != total_bytes:
        data_length, = struct.unpack('<I', data_file.read(4))
        tag_code, = struct.unpack('>H', data_file.read(2))
        image_width, = struct.unpack('<H', data_file.read(2))
        image_height, = struct.unpack('<H', data_file.read(2))
        arc_length = image_width
        if image_width < image_height:
            arc_length = image_height
        temp_image = 255 * np.ones((arc_length, arc_length ,1), np.uint8)
        row_begin = (arc_length - image_height) // 2
        col_begin = (arc_length - image_width) // 2
        for row in range(row_begin, image_height + row_begin):
            for col in range(col_begin, image_width + col_begin):
                temp_image[row, col], = struct.unpack('B', data_file.read(1))
        decoded_bytes += data_length
        result_image = cv2.resize(temp_image, (image_size, image_size))
        if tag_code not in code_map:
            code_map[tag_code] = new_label
            new_label += 1
        image_list.append(result_image)
        label_list.append(code_map[tag_code])

    return image_list, label_list, code_map


if __name__=='__main__':

    if len(sys.argv) < 5:
        sys.stderr.write('Please specify source file, target file, image size and map file \n')
        sys.exit()

    src_file = sys.argv[1]
    dst_file = sys.argv[2]
    image_size = int(sys.argv[3])
    map_file = sys.argv[4]
    gnt2npy(src_file, dst_file, image_size, map_file)