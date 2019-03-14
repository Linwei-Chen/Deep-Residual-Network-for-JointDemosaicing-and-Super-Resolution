import torch
from Test_class import Run_test
from NewResNet import Net

test_data = "/Users/chenlinwei/Desktop/计算机学习资料/TrainData/RAISE_1K/Train_Data.txt"

# 要加上下面这行，不然会出错
torch.set_default_tensor_type('torch.DoubleTensor')

# def Run_test(net, model_path, test_data_path,save_to, as_name):
if 1:
    Run_test(net=Net(2), model_path='./Model_ResNet=2.pkl',
             test_data_path=test_data,
             save_to="./test_result_HD/2-level/", as_name="test_result_PSNR.txt", _block_size=256,
             _test_list=range(0, 10))

else:
    Run_test(net=Net(24), model_path='./Model.pkl',
             test_data_path=test_data,
             save_to="./test_result_HD/", as_name="test_result_PSNR.txt", _block_size=256, _test_list=range(1, 10))
