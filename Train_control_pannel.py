import os
# from Train import *
from PIL import Image
import torch
import tqdm
import torch.utils.data as data
import torch.nn as nn
from torchvision import transforms
import time
import random
from DataSet import CustomDataset
from Model import Net
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from Test_class import *

# 检查GPU是否可用
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("cuda :", torch.cuda.is_available(), "GPU Num : ", torch.cuda.device_count())

BATCH_BLOCK_SIZE = 512

# *** 路径 ***
SAVE_PATH = './Saved_Models/20190226Traned_Model'
MODEL_FILENAME = 'Model.pkl'
MODEL_SAVE_PATH = os.path.join(SAVE_PATH, MODEL_FILENAME)
BEST_MODEL_FILENAME = 'Best_Model.pkl'
BEST_MODEL_SAVE_PATH = os.path.join(SAVE_PATH, BEST_MODEL_FILENAME)
SSIM_BEST_MODEL_FILENAME = 'SSIM_Best_Model.pkl'
SSIM_BEST_MODEL_SAVE_PATH = os.path.join(SAVE_PATH, SSIM_BEST_MODEL_FILENAME)
PARA_FILENAME = 'Para.pkl'
PARA_SAVE_PATH = os.path.join(SAVE_PATH, PARA_FILENAME)
TRAIN_DATA_DIR = os.path.join(os.path.expanduser('~'), 'Dataset/RAISE_8K')
TRAIN_DATA_PATH = os.path.join(TRAIN_DATA_DIR, '8K_TRAIN_DATA/8K_TRAIN_DATA.txt')
TEST_DATA_PATH = "./TEST_DATA/TEST_DATA.txt"


def txt_to_path_list(txt_path):
    with open(txt_path, 'r') as f:
        return list(map(lambda line: line.strip().split(' '), f))


def loading_model(model_path=BEST_MODEL_SAVE_PATH):
    model = Net(resnet_level=24).to(DEVICE)
    try:
        print('===> Loading the saved model...')
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        return model
    except FileNotFoundError:
        print('===> Loading the saved model fail, create a new one...')
        return model
    finally:
        pass


def create_model_imgs(test_list, model_path=BEST_MODEL_SAVE_PATH, save_img=False):
    test_model = loading_model(model_path=model_path)
    PSNR_AVG, SSIM_AVG = 0.0, 0.0
    # img_list = []
    l = len(test_list)
    content = ''
    for i in range(l):  # tqdm(range(l)):
        print('===> [{}/{}]'.format(i, l))
        blks, h, w = padding_and_to_blks(bayer_rgb_img=Image.open(test_list[i][0]).convert('RGB'),
                                         block_size=BATCH_BLOCK_SIZE)
        model_img = run_forward(test_model, DEVICE, blks, h, w, block_size=BATCH_BLOCK_SIZE)
        # img_list.append(model_img)
        PSNR, SSIM = compare(label=Image.open(test_list[i][1]).convert('RGB'), Model_img=model_img)
        model_img_file_name = 'No.{}_PSNR={:.4f}_SSIM={:.4f}.TIF'.format(i + 1, PSNR, SSIM)
        real_img_file_name = 'No.{}_Real.TIF'.format(i + 1)
        with open(file=os.path.join(IMG_SAVE_PATH, 'Result.txt'), mode='a') as f:
            f.write(model_img_file_name + '\n')
        model_img_save_path = os.path.join(IMG_SAVE_PATH, model_img_file_name)
        real_img_save_path = os.path.join(IMG_SAVE_PATH, real_img_file_name)
        if save_img:
            model_img.save(model_img_save_path, format='TIFF')
            Image.open(test_list[i][1]).convert('RGB').save(real_img_save_path, format='TIFF')
        PSNR_AVG += PSNR
        SSIM_AVG += SSIM
    PSNR_AVG /= l
    SSIM_AVG /= l
    print('PSNR_AVG :', PSNR_AVG, 'SSIM_AVG :', SSIM_AVG)
    with open(file=os.path.join(IMG_SAVE_PATH, 'Result.txt'), mode='a') as f:
        f.write('{} Pic:\nPSNR_AVG={:.12f}\nSSIM_AVG={:.12f}\n\n'.format(l, PSNR_AVG, SSIM_AVG))


def show_graph(show_len=200):
    show_len = min(show_len, len(loss_list), len(result_list))
    plt.title('Result Analysis')
    show_range = range(min(show_len, len(result_list)))
    plt.subplot(212)
    plt.plot(show_range, loss_list[-len(show_range):], color='green', label='Loss')
    plt.xlabel('iteration times')
    plt.ylabel('Loss')

    plt.subplot(221)
    plt.plot(show_range,
             [result_list[i + len(result_list) - show_len][0] for i in show_range],
             color='red', label='PSNR')
    plt.plot(show_range, [best_result[0] for i in show_range], color='black', label='PSNR_MAX')
    plt.xlabel('iteration times')
    plt.ylabel('PSNR')

    plt.subplot(222)
    plt.plot(show_range, [best_result[1] for i in show_range], color='black', label='SSIM_MAX')
    plt.plot(show_range,
             [result_list[i + len(result_list) - show_len][1] for i in show_range],
             color='blue', label='SSIM')
    plt.legend()  # 显示图例
    plt.xlabel('iteration times')
    plt.ylabel('SSIM')
    plt.show()


para = torch.load(PARA_SAVE_PATH)
batch_counter = para['batch_counter']
lr = para['lr']
loss_list = para['loss_list']
result_list = para['result_list']
best_result = para['best_result']
hard_cases_list = para['hard_cases_list']

# new_lr = 1e-7
# optimizer = torch.optim.Adam(loading_model().parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
# optimizer.state_dict().update(para['optimizer param_groups'])
# para.update({'lr': new_lr})
# para.update({'optimizer param_groups': optimizer.state_dict()['param_groups'][0]})
# torch.save(para, PARA_SAVE_PATH)

# best_result=[32.4,0.93]
# para.update({'train_dataset_list': []})
# para.update({'hard_cases_list': []})
# torch.save(para, PARA_SAVE_PATH)

# best_result=[32.4,0.93]
# para.update({'best_result': best_result})
# torch.save(para, PARA_SAVE_PATH)

print('batch_counter', batch_counter)
print('lr:', lr)
print('loss_list', loss_list[-3:])
print('result_list', result_list[-3:])
print('best_result', best_result)
# print('hard_cases_list', hard_cases_list)
# show_graph(show_len=350)
# for i in range(42):
#     create_model_imgs(test_list=txt_to_path_list(TEST_DATA_PATH)[i * 50:(i + 1) * 50], model_path=BEST_MODEL_SAVE_PATH)
# create_model_imgs(test_list=txt_to_path_list(TEST_DATA_PATH)[:50], model_path=MODEL_SAVE_PATH)
# create_model_imgs(test_list=txt_to_path_list(TEST_DATA_PATH)[:50], model_path=SSIM_BEST_MODEL_SAVE_PATH)
# create_model_imgs(test_list=txt_to_path_list(FAST_TEST_PATH)[:1000], model_path=MODEL_SAVE_PATH)
# print(os.path.join(IMG_SAVE_PATH, 'No.{}_PSNR={:.4f}_SSIM={:.4f}.TIF'.format(1, 40, 1)))
