import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from PIL import Image
from DataSet import CustomDataset
from NewResNet import Net
from multiprocessing import Process
from Test_class import Run_test

# *** 超参数***        `
Parameter_path = './Final_train_LR.txt'
MODEL_PATH = './Final_Model.pkl'
EPOCH = 1
HALF_LR_STEP = 40000
LR = 0.0001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 训练集与测试集的路径
train_data_path = "../8K_TRAIN_DATA/8K_TRAIN_DATA.txt"
test_data_path = "../8K_CROSS_DATA/8K_CROSS_DATA.txt"
BATCH_BLOCK_SIZE = 64
BATCH_SIZE = 12
DATA_SHUFFLE = True

# 检查GPU是否可用
print("cuda:", torch.cuda.is_available(), "GPUs", torch.cuda.device_count())

# 保存和恢复模型
# https://www.cnblogs.com/nkh222/p/7656623.html
# https://blog.csdn.net/quincuntial/article/details/78045036
#
# 保存
# torch.save(the_model.state_dict(), PATH)
# 恢复
# the_model = TheModelClass(*args, **kwargs)
# the_model.load_state_dict(torch.load(PATH))

# # 只保存网络的参数, 官方推荐的方式
# torch.save(net.state_dict(), 'net_params.pkl')
## 加载网络参数
# net.load_state_dict(torch.load('net_params.pkl'))

print("Loading the LR...")
try:
    P = open(Parameter_path)
    P = list(P)
    LR = float(P[0])
except:
    print("Loading LR fail...")

print("Loading the saving Model...")
MyNet = Net(24).to(device)

try:
    MyNet.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except:
    print("Loading Fail.")
    pass
print("Loading the Training data...")

MyData = CustomDataset(file_path=train_data_path,
                       block_size=BATCH_BLOCK_SIZE)

# CLASS torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
# sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate>,
# pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)

train_data = data.DataLoader(dataset=MyData,
                             batch_size=BATCH_SIZE,
                             shuffle=DATA_SHUFFLE,
                             num_workers= 4,
                             pin_memory=True)

# CLASS torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
Optimizer = torch.optim.Adam(MyNet.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
# CLASS torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
Loss_Func = nn.MSELoss()

counter = 0

print("Start training...")
for epoch in range(EPOCH):
    for step, (data, label) in enumerate(train_data):
        counter = counter + 1
        if counter != 0 and counter % HALF_LR_STEP == 0:
            LR = LR / 2
            Optimizer = torch.optim.Adam(MyNet.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08)
            with open(Parameter_path, 'w') as f:
                f.write(str(LR))
                print('LR:', LR)

        data, label = data.to(device), label.to(device)
        start = time.perf_counter()
        out = MyNet(data)
        # print(type(out), out.shape)
        loss = Loss_Func(out, label)
        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()
        print(loss)
        print(epoch, step)
        print("Time:", time.perf_counter() - start)
        if counter != 0 and 0 == counter % 100:
            print("Saving the model...")
            torch.save(MyNet.state_dict(), MODEL_PATH)
