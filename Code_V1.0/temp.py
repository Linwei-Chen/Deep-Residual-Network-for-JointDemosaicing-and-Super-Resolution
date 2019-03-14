from PIL import Image
import numpy as np
import os
import time
import math
from skimage.measure import compare_ssim, compare_psnr

path = '/Users/chenlinwei/Desktop/计算机学习资料/20181218 Deep Residual Network for Joint Demosaicing and Super-Resolution/Final_test/final'


# a = '/Users/chenlinwei/Desktop/计算机学习资料/20181218 Deep Residual Network for Joint Demosaicing and Super-Resolution/Final_test/final_test99_PSNR=28.5774.TIF'
# b = '/Users/chenlinwei/Desktop/计算机学习资料/20181218 Deep Residual Network for Joint Demosaicing and Super-Resolution/Final_test/final_test99_Real.TIF'


def file_name(path):
    L = []
    for root, dirs, files in os.walk(path):
        # print("root", root)
        # print("dirs", dirs)
        # print("files", type(files), files)
        l = len(files)
        print("len:", l)
        files.sort(key=len)
        for i in range(l):
            file = files[i]
            for j in range(i + 1, l):
                file_2 = files[j]
                if os.path.splitext(file)[1] == '.TIF':
                    if (file[:2] == '._'):
                        file = file[2:]
                    if (file_2[10:file_2[6:].find('_') + 6] == file[10:file[6:].find('_') + 6]):
                        # print(file, file_2)
                        # L.append([os.path.join(root, file), os.path.join(root, file_2)])
                        temp = [file, file_2]
                        temp.sort(key=len)
                        L.append(temp)
                        break
                    # L.append(root+'/'+file)
    # print(L)
    return L


def sort_key(item):
    item = item[0]
    return int(item[10:item[6:].find('_') + 6])


l = file_name(path)
print(l)
l.sort(key=sort_key, reverse=False)
l = l[:100]
SSIM_SUM = 0.0
SSIM_STR = ''
PSNR_STR = ''
for i in l:
    print(i)
    a = os.path.join(path, i[0])
    b = os.path.join(path, i[1])
    a = Image.open(a).convert('RGB')
    b = Image.open(b).convert('RGB')
    a = np.array(a)
    b = np.array(b)
    a = np.array(a)
    b = np.array(b)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a = a[:h, :w]
    b = b[:h, :w]
    # print(a.shape, b.shape)
    SSIM = compare_ssim(X=a, Y=b, full=0, gaussian_weights=0, win_size=11, multichannel=1)
    print(SSIM)
    SSIM_SUM += SSIM
    SSIM_STR += str(SSIM) + '\n'
    PSNR_STR += i[1][-11:-4] + '\n'

print("SSIM_AVG:", SSIM_SUM / len(l))
print(SSIM_STR)
print(PSNR_STR)
