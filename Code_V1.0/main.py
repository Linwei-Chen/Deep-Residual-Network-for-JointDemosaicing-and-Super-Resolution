import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time

from NewResNet import Net
from DataSet import CustomDataset
from multiprocessing import Process
from Test_class import Run_test

# *** 超参数***
BATC_SIZE = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("cuda:", torch.cuda.is_available(),"GPUs",torch.cuda.device_count())

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

print("Loading the saving Model...")
MyNet = Net(24).to(device)
try:
    MyNet.load_state_dict(torch.load('./Model.pkl'))
except:
    print("Loading Fail.")
    pass

# 训练集与测试集的路径
train_data_path = "/Users/chenlinwei/Desktop/计算机学习资料/TrainData/RAISE_1K/Train_Data.txt"
test_data_path = "/Users/chenlinwei/Desktop/计算机学习资料/TrainData/RAISE_1K/Test_Data.txt"
all_data_path = "/Users/chenlinwei/Desktop/计算机学习资料/TrainData/RAISE_1K/Data_Read.txt"

print("Loading the Training data...")
MyData = CustomDataset(train_data_path, random_augment=10, block_size=32)

# CLASS torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
# sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate>,
# pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)

train_data = data.DataLoader(dataset=MyData,
                             batch_size=16,
                             shuffle=True)

# CLASS torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
Optimizer = torch.optim.Adam(MyNet.parameters(), lr=0.0000001, betas=(0.9, 0.999), eps=1e-08)
# CLASS torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
Loss_Func = nn.MSELoss()

EPOCH = 1000000
to_PIL_image = transforms.ToPILImage()


def adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] / 2

counter=0
LR=0.0001
# 8 619
print("Start training...")
for epoch in range(EPOCH):
    for step, (data, label) in enumerate(train_data):
        counter=counter+1
        if(counter!=0 and counter%10000==0):
            LR=LR/2
        Optimizer = torch.optim.Adam(MyNet.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08)
        # print(type(data), type(label))
        # print(data.shape, label.shape)
        # print(data, label)
        # for i in range(data.size()[0]):
        # img1 = to_PIL_image(data)
        # img2 = to_PIL_image(label)
        # img1.show()
        # img2.show()
        data, label=data.to(device), label.to(device)
        start = time.process_time()
        out = MyNet(data)
        # print(type(out), out.shape)
        loss = Loss_Func(out, label)
        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()
        print(loss)
        print(epoch, step)
        print("Time:", time.process_time() - start)
        if counter != 0 and 0 == counter % 10:
            print("Saving the model...")
            torch.save(MyNet.state_dict(), './Model.pkl')
        if counter != 0 and 0 == counter % 99:
            torch.set_default_tensor_type('torch.DoubleTensor')
            # def Run_test(net, model_path, test_data_path, save_to, as_name, _block_size=32):
            multi_Process = Process(target=Run_test(net=Net(24), model_path='./Model.pkl',
                                                    test_data_path="/Users/chenlinwei/Desktop/计算机学习资料/TrainData/RAISE_1K/Fast_Test_Data.txt",
                                                    save_to="./test_result/", as_name="test_result_PSNR.txt",
                                                    _block_size=64))
            multi_Process.start()
