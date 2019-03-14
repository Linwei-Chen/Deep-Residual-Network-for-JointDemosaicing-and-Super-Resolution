# -*- coding: utf-8 -*-

import os
import math
import numpy as np
from PIL import Image
import time

SOURCE_PATH = './8K_CROSS_DATA'  # "/Volumes/750GB/RAISE"
BLOCK_SIZE = 128  # Ground Truth 图片大小
TRAIN_DATA_SAVE_PATH = "./8K_TRAIN_DATA"
CROSS_DATA_SAVE_PATH = "./8K_CROSS_DATA"
TEST_DATA_SAVE_PATH = "./8K_TEST_DATA"


def DownSize(img):
    for i in range(3):
        w, h = img.size
        # print(int(w / 1.25), int(h / 1.25))
        img = img.resize((int(w / 1.25), int(h / 1.25)), Image.ANTIALIAS)
    return img


def To_Bayer(img):
    w, h = img.size
    img = img.resize((int(w / 2), int(h / 2)), Image.ANTIALIAS)
    w, h = img.size
    # r,g,b=img.split()
    data = np.array(img)
    """
    R G R G
    G B G B
    R G R G
    G B G B
    """
    bayer_mono = np.zeros((h, w))
    for r in range(h):
        for c in range(w):
            if (0 == r % 2):
                if (1 == c % 2):
                    data[r, c, 0] = 0
                    data[r, c, 2] = 0

                    bayer_mono[r, c] = data[r, c, 1]
                else:
                    data[r, c, 1] = 0
                    data[r, c, 2] = 0

                    bayer_mono[r, c] = data[r, c, 0]
            else:
                if (0 == c % 2):
                    data[r, c, 0] = 0
                    data[r, c, 2] = 0

                    bayer_mono[r, c] = data[r, c, 1]
                else:
                    data[r, c, 0] = 0
                    data[r, c, 1] = 0

                    bayer_mono[r, c] = data[r, c, 2]

    # 三通道Bayer图像
    bayer = Image.fromarray(data)
    # bayer.show()

    return bayer


# 获取文件夹下文件的方法
# https://blog.csdn.net/LZGS_4/article/details/50371030
# files = os.listdir(source_path)
# print(files)
# https://blog.csdn.net/lsq2902101015/article/details/51305825

def file_name(path):
    L = []
    for root, dirs, files in os.walk(path):
        print("root", root)
        print("dirs", dirs)
        print("files", type(files), files)
        for file in files:
            if os.path.splitext(file)[1] == '.TIF':
                if (file[:2] == '._'):
                    file = file[2:]
                L.append(os.path.join(root, file))
                # L.append(root+'/'+file)
    return L


img_list = file_name(SOURCE_PATH)

img_list_size = len(img_list)
train_data = img_list[:math.ceil(img_list_size * 0.9)]
test_data = img_list[math.floor(img_list_size * 0.9):]

print('train data size:', len(train_data))
print('test data size:', len(test_data))


def get_name(path):
    print(type(path))
    size = len(path)
    return path[size - 14:size - 4]


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)


def train_process(img_path, save_path):
    txt = open(save_path + '/' + save_path + '.txt', 'a')
    time_start = time.perf_counter()
    counter = 0
    for No, i in enumerate(img_path):
        print(i)
        img = Image.open(i)
        img = DownSize(img)
        # img.show()
        w, h = img.size
        row = int(h / BLOCK_SIZE)
        col = int(w / BLOCK_SIZE)
        print("No.", counter, "WxH=", w, 'x', h)
        counter = counter + 1
        img_ID = 0
        name = get_name(i)
        for r in range(row):
            for c in range(col):
                # print('img_ID:', img_ID)
                # 创建空图块并填充，保存原图并且
                # temp = Image.new(mode='RGB', size=(BLOCK_SIZE, BLOCK_SIZE))
                temp = img.crop((r * BLOCK_SIZE, c * BLOCK_SIZE, (r + 1) * BLOCK_SIZE, (c + 1) * BLOCK_SIZE)).convert(
                    'RGB')
                # temp.show()
                temp_bayer = To_Bayer(temp)

                mkdir(save_path + '/' + name)
                # 生成数据以及标签的路径信息,保存于txt文件中，并按路径保存图像
                data_str = save_path + '/' + name + '/' + name + '_' + str(img_ID) + 'data.TIF'
                label_str = save_path + '/' + name + '/' + name + '_' + str(img_ID) + 'label.TIF'
                txt.write(data_str + ' ' + label_str + '\n')

                temp.save(label_str, 'TIFF')
                temp_bayer.save(data_str, 'TIFF')

                img_ID = img_ID + 1
                time_used = time.perf_counter() - time_start
        print('Time used:', time_used)
        print('Time remain:', time_used / (No + 1) * len(img_path) - time_used)


L = [2, 4, 5, 7, 9, 10, 12, 15, 17, 18, 19, 20, 21, 25, 29, 30, 32, 33, 34, 36, 37, 38, 42, 43, 45, 48, 52, 53, 54, 56,
     60, 61, 62, 65, 66, 67, 70, 71, 72, 75, 77, 79, 81, 83, 86, 87, 90, 91, 96, 99]


def test_process(img_path, save_path, img_ID=870):
    txt = open(save_path + '/' + save_path + '.txt', 'a')
    time_start = time.perf_counter()
    counter = 0
    for No, i in enumerate(img_path):
        # if No + 1 not in L: continue

        print("No.", counter)
        counter = counter + 1

        data_str = save_path + '/' + str(img_ID) + 'data.TIF'
        label_str = save_path + '/' + str(img_ID) + 'label.TIF'
        txt.write(data_str + ' ' + label_str + '\n')

        temp = Image.open(i)
        for j in range(1):
            temp = DownSize(temp)
        temp_bayer = To_Bayer(temp)
        temp.save(label_str, 'TIFF')
        temp_bayer.save(data_str, 'TIFF')

        img_ID = img_ID + 1
        time_used = time.perf_counter() - time_start
        print('Time used:', time_used)
        print('Time remain:', time_used / (No + 1) * len(img_path) - time_used)

    txt.close()


def resave(img_list, save_path):
    time_start = time.perf_counter()
    for i ,f in enumerate(img_list,1):
        Image.open(f).convert('RGB').save(os.path.join(save_path, os.path.split(f)[1]), format='TIFF')
        time_used = time.perf_counter() - time_start
        print('===> No.{} Pic used:{}'.format(i, time_used))
        print('Time remain:', time_used / i * len(img_list) - time_used)


# train_process(train_data[:1], TRAIN_DATA_SAVE_PATH)

# test_process(test_data[:1], TEST_DATA_SAVE_PATH)

# train_process(img_list[:6000], TRAIN_DATA_SAVE_PATH)
test_process(img_list[:], TEST_DATA_SAVE_PATH, img_ID=0)
# test_process(img_list[:], CROSS_DATA_SAVE_PATH)
# resave(img_list=img_list[6000:], save_path=CROSS_DATA_SAVE_PATH)
