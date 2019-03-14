import torch
from PIL import Image
import numpy as np
from Test_class import LR_to_HR
from NewResNet import Net
import random

torch.set_default_tensor_type('torch.DoubleTensor')
img_path = "/Users/chenlinwei/Desktop/2.jpg"
MODEL_PATH = './Final_Model.pkl'
# def LR_to_HR(net, model_path, img_path, save_to, _block_size=32):

if 1:
    LR_to_HR(net=Net(24), model_path=MODEL_PATH,
             img_path=img_path,
             save_to="./" + str(random.random()), _block_size=64 )
