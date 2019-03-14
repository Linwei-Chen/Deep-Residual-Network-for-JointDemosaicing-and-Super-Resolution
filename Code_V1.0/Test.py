import torch
from Test_class import Run_test
from NewResNet import Net

test_data = "./TEST_DATA/TEST_DATA.txt"
cross_test_data = "./8K_CROSS_DATA/8K_CROSS_DATA_10PIC.txt"
test_data_128x128 = "/Users/chenlinwei/Desktop/计算机学习资料/TrainData/RAISE_1K/Fast_Test_Data.txt"
MODEL_PATH = './Final_Model.pkl'
HD_train_data = '/Users/chenlinwei/Desktop/计算机学习资料/TrainData/RAISE_1K/Train_Data.txt'
final_test = "./8K_TEST_DATA/8K_TEST_DATA.txt"

# 要加上下面这行，不然会出错
torch.set_default_tensor_type('torch.DoubleTensor')

# def Run_test(net, model_path, test_data_path,save_to, as_name):
if 1:
    Run_test(net=Net(24), model_path=MODEL_PATH,
             test_data_path=test_data_128x128,
             save_to="./test_result/", as_name="Final_Model", _block_size=64)

if 0:
    Run_test(net=Net(24), model_path=MODEL_PATH,
             test_data_path=test_data,
             save_to="./test_result/", as_name="Final_Model_HD", _block_size=64)
if 0:
    Run_test(net=Net(24), model_path=MODEL_PATH,
             test_data_path=cross_test_data,
             save_to="./test_result/", as_name="cross_test", _block_size=64,
             _test_list=range(0, 10))
if 0:
    Run_test(net=Net(24), model_path=MODEL_PATH,
             test_data_path=final_test,
             save_to="./Final_test/", as_name="final_test", _block_size=64,
             _test_list=range(0, 50))

if 0:
    Run_test(net=Net(24), model_path=MODEL_PATH,
             test_data_path=HD_train_data,
             save_to="./test_result_HD/", as_name="HD_train_data", _block_size=64,
             _test_list=range(0, 10))
