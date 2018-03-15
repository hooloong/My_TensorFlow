import os
import cv2
import numpy as np

# import os
#  汇总到一个新的目录
# old_dir = 'images'
# new_dir = 'girls'
# if not os.path.exists(new_dir):
#     os.makedirs(new_dir)
#
# count = 0
# for (dirpath, dirnames, filenames) in os.walk(old_dir):
#     for filename in filenames:
#         if filename.endswith('.jpg'):
#             new_filename = str(count) + '.jpg'
#             os.rename(os.sep.join([dirpath, filename]), os.sep.join([new_dir, new_filename]))
#             print(os.sep.join([dirpath, filename]))
#             count += 1
# print("Total Picture: ", count)

# 缩小到64x64像素
# image_dir = 'girls'
# new_girl_dir = 'little_girls'
# if not os.path.exists(new_girl_dir):
#     os.makedirs(new_girl_dir)
#
# for img_file in os.listdir(image_dir):
#     img_file_path = os.path.join(image_dir, img_file)
#     img = cv2.imread(img_file_path)
#     if img is None:
#         print("image read fail")
#         continue
#     height, weight, channel = img.shape
#     if height < 200 or weight < 200 or channel != 3:
#         continue
#     # 你也可以转为灰度图片(channel=1)，加快训练速度
#     # 把图片缩放为64x64
#     img = cv2.resize(img, (64, 64))
#     new_file = os.path.join(new_girl_dir, img_file)
#     cv2.imwrite(new_file, img)
#     print(new_file)

# 判断两张图片是否完全一样（使用哈希应该要快很多）
def is_same_image(img_file1, img_file2):
    img1 = cv2.imread(img_file1)
    img2 = cv2.imread(img_file2)
    if img1 is None or img2 is None:
        return False
    if img1.shape == img2.shape and not (np.bitwise_xor(img1, img2).any()):
        return True
    else:
        return False


# 去除重复图片
file_list = os.listdir('little_girls')
try:
    for img1 in file_list:
        print(len(file_list))
        for img2 in file_list:
            if img1 != img2:
                if is_same_image('little_girls/' + img1, 'little_girls/' + img2) is True:
                    print(img1, img2)
                    os.remove('little_girls/' + img1)
        file_list.remove(img1)
except Exception as e:
    print(e)