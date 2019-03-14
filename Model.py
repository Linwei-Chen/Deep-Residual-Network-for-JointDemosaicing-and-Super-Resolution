import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np


# ResNet
# https://blog.csdn.net/sunqiande88/article/details/80100891
class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.shortcut = nn.Sequential()
        self.active_f = nn.PReLU()

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.active_f(out)
        return out


class Net(nn.Module):

    def __init__(self, resnet_level=24):
        super(Net, self).__init__()

        # ***Stage1***
        # class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.stage1_1_conv4x4 = nn.Conv2d(in_channels=1, out_channels=256,
                                          kernel_size=4, stride=2, padding=1, bias=True)
        # Reference:
        # CLASS torch.nn.PixelShuffle(upscale_factor)
        # Examples:
        #
        # >>> pixel_shuffle = nn.PixelShuffle(3)
        # >>> input = torch.randn(1, 9, 4, 4)
        # >>> output = pixel_shuffle(input)
        # >>> print(output.size())
        # torch.Size([1, 1, 12, 12])

        self.stage1_2_SP_conv = nn.PixelShuffle(2)
        self.stage1_2_conv4x4 = nn.Conv2d(in_channels=64, out_channels=256,
                                          kernel_size=3, stride=1, padding=1, bias=True)

        # CLASS torch.nn.PReLU(num_parameters=1, init=0.25)
        self.stage1_2_PReLU = nn.PReLU()

        # ***Stage2***
        self.stage2_ResNetBlock = []
        for i in range(resnet_level):
            self.stage2_ResNetBlock.append(ResidualBlock())
        self.stage2_ResNetBlock = nn.Sequential(*self.stage2_ResNetBlock)

        # ***Stage3***
        self.stage3_1_SP_conv = nn.PixelShuffle(2)
        self.stage3_2_conv3x3 = nn.Conv2d(in_channels=64, out_channels=256,
                                          kernel_size=3, stride=1, padding=1, bias=True)
        self.stage3_2_PReLU = nn.PReLU()
        self.stage3_3_conv3x3 = nn.Conv2d(in_channels=256, out_channels=3,
                                          kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        out = self.stage1_1_conv4x4(x)
        out = self.stage1_2_SP_conv(out)
        out = self.stage1_2_conv4x4(out)
        out = self.stage1_2_PReLU(out)

        out = self.stage2_ResNetBlock(out)

        out = self.stage3_1_SP_conv(out)
        out = self.stage3_2_conv3x3(out)
        out = self.stage3_2_PReLU(out)
        out = self.stage3_3_conv3x3(out)

        return out
