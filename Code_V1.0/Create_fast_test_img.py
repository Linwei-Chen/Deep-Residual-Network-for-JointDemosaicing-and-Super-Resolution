from PIL import Image
import numpy as np


# https://www.jianshu.com/p/85eba2a51142
# PIL.Image.NEAREST：最低质量
# PIL.Image.BILINEAR：双线性
# PIL.Image.BICUBIC：三次样条插值
# PIL.Image.ANTIALIAS：最高质量

def DownSize_256x256(img):
    for i in range(6):
        w, h = img.size
        # print(int(w / 1.25), int(h / 1.25))
        img = img.resize((int(w / 1.25), int(h / 1.25)), Image.ANTIALIAS)

    min_size = min(w/2, h/2)
    img = img.crop((0, 0, min_size, min_size))
    img = img.resize((256, 256), Image.ANTIALIAS)
    return img


def DownSize_2(img):
    pass


def To_Bayer(img):
    w, h = img.size
    img = img.resize((int(w / 2), int(h / 2)), Image.ANTIALIAS)
    w, h = img.size
    # r,g,b=img.split()
    data = np.array(img)
    """
    R G R G
    G B G B
    R G R G
    G B G B
    """
    bayer_mono = np.zeros((h, w))
    for r in range(h):
        for c in range(w):
            if (0 == r % 2):
                if (1 == c % 2):
                    data[r, c, 0] = 0
                    data[r, c, 2] = 0

                    bayer_mono[r, c] = data[r, c, 1]
                else:
                    data[r, c, 1] = 0
                    data[r, c, 2] = 0

                    bayer_mono[r, c] = data[r, c, 0]
            else:
                if (0 == c % 2):
                    data[r, c, 0] = 0
                    data[r, c, 2] = 0

                    bayer_mono[r, c] = data[r, c, 1]
                else:
                    data[r, c, 0] = 0
                    data[r, c, 1] = 0

                    bayer_mono[r, c] = data[r, c, 2]

    # 三通道Bayer图像
    bayer = Image.fromarray(data)
    # bayer.show()

    return bayer


resize_dir = "/Users/linweichen/Desktop/计算机学习资料/TrainData/RAISE_1K/Resize/"
save_path = "/Users/linweichen/Desktop/计算机学习资料/TrainData/RAISE_1K/Fast_test/"

for i in range(1, 1001):
    img = Image.open(resize_dir + "/real" + str(i) + ".TIF", "r")
    # img.show()
    img = DownSize_256x256(img)
    # img.show()
    img.save(save_path + str(i) + "real" + ".TIF", "TIFF")

    bayer=To_Bayer(img)
    bayer.save(save_path + str(i) + "bayer" + ".TIF", "TIFF")
