import os
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
import torch.backends.cudnn as cudnn
from Test_class import *

# *** 超参数***        `

EPOCH = 1
SUB_EPOCH_SIZE = 200
SUB_EPOCH = 10000
BATCH_COUNTER = 0
LR_HALF = 10000
LR = 0.0001
SEED = 666
BATCH_BLOCK_SIZE = 64
BATCH_SIZE = 12
DATA_SHUFFLE = True
NUM_WORKERS = 2
# 检查GPU是否可用
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("cuda :", torch.cuda.is_available(), "GPU Num : ", torch.cuda.device_count())
if torch.cuda.is_available():
    # cudnn.benchmark = True
    cudnn.deteministic = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
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


# *** 模型、数据集 ***


def init_para():
    start_time = time.perf_counter()
    try:
        print('===> Find the para_saved file')
        open(PARA_SAVE_PATH)
    except FileNotFoundError:
        print('===> The para_saved file Not exist, creating new one...')
        train_dataset_list = txt_to_path_list(TRAIN_DATA_PATH)
        random.shuffle(train_dataset_list)
        torch.save({
            'epoch': EPOCH,
            'batch_counter': BATCH_COUNTER,
            'lr': LR,
            'optimizer param_groups': torch.optim.Adam(Net().to(DEVICE).parameters(),
                                                       lr=LR,
                                                       betas=(0.9, 0.999),
                                                       eps=1e-08,
                                                       amsgrad=True).state_dict()['param_groups'][0],
            'train_dataset_list': train_dataset_list,
            'loss_list': [],
            'result_list': [],
            'hard_cases_list': [],
            'best_result': [0, 0]
        }, PARA_SAVE_PATH)
        print('==> Done with initialization!')
    finally:
        print('===> Init_para used time: ', time.perf_counter() - start_time)


def txt_to_path_list(txt_path):
    with open(txt_path, 'r') as f:
        return list(map(lambda line: line.strip().split(' '), f))


def get_train_dataset():
    start_time = time.perf_counter()
    try:
        print('===> Try to get train dataset from saved saved file...')
        train_dataset_list = para['train_dataset_list']
        epoch = para['epoch']
        print('===> Pre train_dataset_list : ', len(train_dataset_list))
        print('===> Epoch :', epoch)
        # random.shuffle(train_dataset_list)
        L = min(SUB_EPOCH_SIZE * BATCH_SIZE, len(train_dataset_list))
        # print(len(para['hard_cases_list']))
        if L <= 0:
            if len(para['hard_cases_list']) > 0:
                print('===> Loading hard_cases_list...')
                global hard_cases_list
                train_dataset_list = hard_cases_list  # para[' hard_cases_list']
                hard_cases_list = []
            else:
                print('===> Loading TXT...')
                train_dataset_list = txt_to_path_list(TRAIN_DATA_PATH)

            epoch = epoch + 1

        L = min(SUB_EPOCH_SIZE * BATCH_SIZE, len(train_dataset_list))
        if L <= 0:
            raise (FileNotFoundError('Train_data_path.txt File not found'))

        train_dataset_rest_list = train_dataset_list[L:]
        train_dataset_list = train_dataset_list[:L]
        # para.update({'train_dataset_list': train_dataset_rest_list})
        print('===> train_dataset_list now : ', len(train_dataset_list))
        print('===> train_dataset_rest_list: ', len(train_dataset_rest_list))
        # torch.save(para, PARA_SAVE_PATH)
        # print(len(torch.load(PARA_SAVE_PATH)['train_dataset_list']))

        #  DataLoader(dataset, batch_size=1,
        #  shuffle=False, sampler=None,
        #  batch_sampler=None, num_workers=0,
        #  collate_fn=<function default_collate>,
        #  pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None
        return DataLoader(dataset=CustomDataset(data_dir=TRAIN_DATA_DIR, file_path_list=train_dataset_list),
                          batch_size=BATCH_SIZE,
                          shuffle=DATA_SHUFFLE,
                          num_workers=NUM_WORKERS,
                          pin_memory=True), {'train_dataset_list': train_dataset_rest_list, 'epoch': epoch}
    except FileNotFoundError:
        raise FileNotFoundError('File not found')
    finally:
        print('===> Get_train_dataset used time: ', time.perf_counter() - start_time)


def get_test_dataset():
    pass


def loading_model():
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    model = Net(resnet_level=24).to(DEVICE)
    try:
        print('===> Loading the saved model...')
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        return model
    except FileNotFoundError:
        print('===> Loading the saved model fail, create a new one...')
        return model
    finally:
        pass


def train(sub_epoch):
    sub_epoch_loss = 0
    global lr
    global batch_counter
    sub_epoch_start_time = time.perf_counter()
    for iteration, (data, label, path) in enumerate(train_dataset, 1):
        batch_counter = batch_counter + 1
        if batch_counter != 0 and 0 == batch_counter % LR_HALF:
            lr = lr / 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('===> Optimizer update : ', optimizer.state_dict()['param_groups'][0])
        start_time = time.perf_counter()
        data, label = data.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(data), label)
        sub_epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        global loss_list
        if len(loss_list) > 0 and loss.item() > sum(loss_list[max(0, len(loss_list) - 100):]) / 100:
            for i in range(BATCH_SIZE):
                hard_cases_list.append([path[0][i], path[1][i]])
            # print('==> Add to hard_cases_list', hard_cases_list[-8:])
            print('hard_cases_list size: ', len(hard_cases_list))

        print("===> Sub_epoch[{}]({}/{}): Loss: {:.12f}".format(sub_epoch, iteration, len(train_dataset), loss.item()))
        print('No.{} batches'.format(batch_counter), 'Time used :', time.perf_counter() - start_time)

    print("===> Sub_epoch {} Complete: Avg. Loss: {:.12f}".format(sub_epoch, sub_epoch_loss / len(train_dataset)))
    print('{} Batches time used :'.format(len(train_dataset)), time.perf_counter() - sub_epoch_start_time)
    return sub_epoch_loss / len(train_dataset)


def test(model):
    print('===> Testing the performance of model...')
    test_model = model
    test_list = txt_to_path_list(TEST_DATA_PATH)
    PSNR_AVG, SSIM_AVG = 0.0, 0.0
    l = 10
    img_list = []
    for i in range(l):
        # print(test_list[i][0])
        blks, h, w = padding_and_to_blks(bayer_rgb_img=Image.open(test_list[i][0]).convert('RGB'),
                                         block_size=BATCH_BLOCK_SIZE)
        model_img = run_forward(test_model, DEVICE, blks, h, w, block_size=BATCH_BLOCK_SIZE)
        img_list.append(model_img)
        PSNR, SSIM = compare(label=Image.open(test_list[i][1]).convert('RGB'), Model_img=model_img)
        PSNR_AVG += PSNR
        SSIM_AVG += SSIM
    PSNR_AVG /= l
    SSIM_AVG /= l
    global best_result
    global result_list
    result_list.append([PSNR_AVG, SSIM_AVG])
    print('PSNR_AVG :', PSNR_AVG, 'SSIM_AVG :', SSIM_AVG)
    if PSNR_AVG > best_result[0] and SSIM_AVG > best_result[1]:
        best_result = [PSNR_AVG, SSIM_AVG]
        print('*** Saving the best model...')
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'Best_Model_Temp.pkl'))
        os.remove(BEST_MODEL_SAVE_PATH)
        os.rename(os.path.join(SAVE_PATH, 'Best_Model_Temp.pkl'), BEST_MODEL_SAVE_PATH)

        for i in img_list:
            i.show()
    '''
    elif SSIM_AVG > best_result[1]:
        print('*** Saving the SSIM best model...')
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'SSIM_Best_Model_Temp.pkl'))
        os.remove(SSIM_BEST_MODEL_SAVE_PATH)
        os.rename(os.path.join(SAVE_PATH, 'SSIM_Best_Model_Temp.pkl'), SSIM_BEST_MODEL_SAVE_PATH)
    '''


def check_point():
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'Model_Temp.pkl'))
    os.remove(MODEL_SAVE_PATH)
    os.rename(os.path.join(SAVE_PATH, 'Model_Temp.pkl'), MODEL_SAVE_PATH)
    global lr
    global hard_cases_list
    global best_result
    global result_list
    para.update(para_update)
    para.update({
        'batch_counter': batch_counter,
        'lr': lr,
        'optimizer param_groups': optimizer.state_dict()['param_groups'][0],
        'hard_cases_list': hard_cases_list,
        'loss_list': loss_list,
        'result_list': result_list,
        'best_result': best_result
    })
    torch.save(para, os.path.join(SAVE_PATH, 'Para_Temp.pkl'))
    os.remove(PARA_SAVE_PATH)
    os.rename(os.path.join(SAVE_PATH, 'Para_Temp.pkl'), PARA_SAVE_PATH)
    print('Rest list: ', len(para['train_dataset_list']))
    print('Loss list: ', len(para['loss_list']))
    print('Hard_cases_list', len(para['hard_cases_list']))




# if __name__ == '__main__':
print('Start training...')
init_para()
model = loading_model()
para = torch.load(PARA_SAVE_PATH)

batch_counter = para['batch_counter']
lr = para['lr']
loss_list = para['loss_list']
result_list = para['result_list']
best_result = para['best_result']
hard_cases_list = para['hard_cases_list']

optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
optimizer.state_dict().update(para['optimizer param_groups'])
print('===> Optimizer param_groups state: ', optimizer.state_dict()['param_groups'][0])
criterion = nn.MSELoss()

test(loading_model())

for i in range(1, SUB_EPOCH + 1):
    print('===> Batch_counter : ', batch_counter)
    train_dataset, para_update = get_train_dataset()
    new_avg_loss = train(i)
    loss_list.append(new_avg_loss)
    check_point()
    test(model)
