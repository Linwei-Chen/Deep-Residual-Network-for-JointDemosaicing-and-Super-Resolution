import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import random
import numpy as np


# Reference link:
# 如何构建数据集
# https://oidiotlin.com/create-custom-dataset-in-pytorch/
# https://www.pytorchtutorial.com/pytorch-custom-dataset-examples/

# transforms 函数的使用
# https://www.jianshu.com/p/13e31d619c15
# ToTensor：convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W) in the range [0.0,1.0]

# torch.set_default_tensor_type('torch.DoubleTensor')
class CustomDataset(data.Dataset):
    # file_path TXT文件路径
    # random_augment=1 随机裁剪数据增强
    # block_size=64 裁剪大小
    def __init__(self, file_path, block_size=64):
        with open(file_path, 'r') as file:
            self.imgs = list(map(lambda line: line.strip().split(' '), file))
            self.Block_size = block_size
            print("DataSet Size is: ", self.__len__())
            # print(len(self.imgs))
            # for i in self.imgs:
            #     print(len(i))

    def __getitem__(self, index):
        # 注意！！！ 读入的Bayer图像最左上为：
        # R G
        # G B
        # Reference API
        # class torchvision.transforms.RandomCrop(size, padding=0, pad_if_needed=False)
        # class torchvision.transforms.Compose([transforms_list,])->生成一个函数
        data_path, label_path = self.imgs[index]
        # print(index, data_path, label_path)

        data = Image.open(data_path).convert('L')
        label = Image.open(label_path).convert('RGB')

        '''
        # 生成截取图块
        w, h = data.size
        block_size = self.Block_size

        # 图块左上角坐标，为了保证Bayer阵列，必须取偶数
        random_w_pos = random.randint(0, int((w - block_size) / 2)) * 2
        random_h_pos = random.randint(0, int((h - block_size) / 2)) * 2

        # print("w=", w, "h=", h, "Crop at: ", random_w_pos, random_h_pos)
        # Image.crop(box=None)
        # Returns a rectangular region from this image. The box is a 4-tuple defining the left, upper, right, and lower pixel coordinate.
        # box – The crop rectangle, as a (left, upper, right, lower)-tuple.

        # (random_w_pos:random_w_pos + 32, random_h_pos: random_h_pos + 32)

        # 转换成Tensor之前 一定要convert一下
        # !!! https://discuss.pytorch.org/t/runtimeerror-invalid-argument-0/17919/5
        data = data.crop((random_w_pos, random_h_pos,
                              random_w_pos + block_size, random_h_pos + block_size)).convert('L')

        # label[2 * random_w_pos:2 * random_w_pos + 64, 2 * random_h_pos:2 * random_h_pos + 64]
        label = label.crop((2 * random_w_pos, 2 * random_h_pos,
                                2 * random_w_pos + 2 * block_size, 2 * random_h_pos + 2 * block_size)).convert('RGB')
        '''
        trans = transforms.Compose([transforms.ToTensor()])

        data_img = trans(data)
        label_img = trans(label)

        return data_img, label_img

    def __len__(self):
        return len(self.imgs)
