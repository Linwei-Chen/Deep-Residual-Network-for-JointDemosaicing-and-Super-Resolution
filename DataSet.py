import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
import os


def is_tif_file(filename):
    return any(filename.endswith(extension) for extension in [".TIF"])


def bayer2mono(bayer_img):
    bayer_img = np.array(bayer_img)
    bayer_mono = np.max(bayer_img, 2)
    return bayer_mono[:, :, np.newaxis]


# bayer2mono()


class CustomDataset(data.Dataset):
    def __init__(self, data_dir, file_path_list):
        self.imgs = file_path_list
        self.data_dir = data_dir
        print('Dataset size is : ', len(self.imgs))

    def __getitem__(self, index):
        # 注意！！！ 读入的Bayer图像最左上为：
        # R G
        # G B
        data_path, label_path = [os.path.join(self.data_dir, i) for i in self.imgs[index]]
        # print(index, data_path, label_path)

        data = bayer2mono(Image.open(data_path).convert('RGB'))
        label = Image.open(label_path).convert('RGB')

        trans = transforms.Compose([transforms.ToTensor()])

        data_img = trans(data)
        label_img = trans(label)

        return data_img, label_img, self.imgs[index]

    def __len__(self):
        return len(self.imgs)
