import torch
import torchvision
from PIL import Image
import numpy as np
import time
import math
from skimage.measure import compare_ssim, compare_psnr
import torchvision.transforms as transforms
import torch.utils.data as data

PADDING = 2
FORWORD_BLOCK_NUM = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("cuda:", torch.cuda.is_available(), "GPUs", torch.cuda.device_count())


class TestDataset(data.Dataset):
    def __init__(self, img_blks):
        self.img_blks = torch.from_numpy(img_blks / 255.).double()

    def __len__(self):
        return self.img_blks.shape[0]

    def __getitem__(self, index):
        return self.img_blks[index]


class Test:
    def __init__(self, file_path, Net, block_size=64):
        self.Net = Net.to(device)
        self.block_size = block_size
        if file_path != '':
            with open(file_path, 'r') as file:
                self.imgs = list(map(lambda line: line.strip().split(' '), file))

    def padding_and_img_to_blks(self, raw_img):
        # *** step1: 图像预处理
        # s为padding大小
        s = PADDING
        block_size = self.block_size
        # print("block_size:", block_size)
        # print("raw_img.shape: ",raw_img.shape)
        w, h = raw_img.size
        print(w, h)
        raw_data_arr = np.array(raw_img)
        print("raw_data_arr shape", raw_data_arr.shape)

        # 将图片分割成块，向上取整
        row_block = math.ceil(h / block_size)
        col_block = math.ceil(w / block_size)
        block_total = row_block * col_block

        # 根据取整情况，对原图进行镜像Padding
        raw_data_arr = np.pad(raw_data_arr, ((s, row_block * block_size - h + s),
                                             (s, col_block * block_size - w + s)), mode='reflect')

        # *** step2: 将原图切割成块保存于blks
        blks = np.zeros((block_total, 1, block_size + 2 * s, block_size + 2 * s))
        blk_counter = 0
        for i in range(row_block):
            for j in range(col_block):
                r = i * block_size
                c = j * block_size
                blks[blk_counter, :, :, :] = raw_data_arr[r: r + block_size + 2 * s, c:c + block_size + 2 * s]
                blk_counter = blk_counter + 1
        return blks, h, w

    def run_forward(self, blks, h, w):
        block_size = self.block_size
        row_block = math.ceil(h / block_size)
        col_block = math.ceil(w / block_size)
        s = PADDING
        blk_num = blks.shape[0]

        res = np.zeros((blk_num, 3, (block_size + 2 * s) * 2, (block_size + 2 * s) * 2))
        res = torch.from_numpy(res)  # .to(device)

        '''
        img_dataset = TestDataset(blks)
        img_data = data.DataLoader(dataset=img_dataset,
                                   batch_size=FORWORD_BLOCK_NUM,
                                   shuffle=False)

        
        l = range(math.ceil(blk_num / FORWORD_BLOCK_NUM))
        start = time.perf_counter()
        for i, img_blks in enumerate(img_data):
            blk_No = min((l[i] + 1) * FORWORD_BLOCK_NUM, blk_num)
            img_blks = img_blks.to(device)
            out = self.Net(img_blks).detach()
            used_time = time.perf_counter() - start
            print(used_time)
            res[i * FORWORD_BLOCK_NUM:min((l[i] + 1) * FORWORD_BLOCK_NUM, blk_num), :, :] = out
            used_time = time.perf_counter() - start
            print("Processing:", blk_No, '/', blk_num)
            print("Now:", used_time)
            print("Remain:", used_time / blk_No * blk_num - used_time)
        '''
        start = time.perf_counter()
        # blks = torch.from_numpy(blks / 255.).double().to(device)
        with torch.no_grad():
            for i in range(math.ceil(blk_num / FORWORD_BLOCK_NUM)):
                blk_No = min((i + 1) * FORWORD_BLOCK_NUM, blk_num)
                temp = blks[i * FORWORD_BLOCK_NUM:blk_No, :, :, :]
                temp = torch.from_numpy(temp / 255.).to(device)
                temp = self.Net(temp).detach()
                res[i * FORWORD_BLOCK_NUM:min((i + 1) * FORWORD_BLOCK_NUM, blk_num), :, :] = temp.cpu()

                # 输出时间
                used_time = time.perf_counter() - start
                print("Processing:", blk_No, '/', blk_num)
                print("Now:", used_time)
                print("Remain:", used_time / blk_No * blk_num - used_time)

        # 计算完成，将图块拼回图像
        print("BLOCKS:", blk_num)
        start = time.perf_counter()
        res_img = np.zeros((3, 2 * h, 2 * w))
        res = res.cpu().numpy() * 255.0
        # res = res.clip(0, 255)
        print('Copy use:', time.perf_counter() - start)
        # res_img = torch.from_numpy(res_img).double().to(device)
        for i in range(row_block):
            for j in range(col_block):
                r = i * block_size
                c = j * block_size
                block_h = min(h, r + block_size) - r
                block_w = min(w, c + block_size) - c
                res_img[:, 2 * r: 2 * (r + block_h), 2 * c:2 * (c + block_w)] = \
                    res[i * col_block + j, :, 2 * s:2 * (block_h + s), 2 * s:2 * (block_w + s)]
        print('Re-range use:', time.perf_counter() - start)

        # res_img = Image.fromarray((H,W,3))
        # 将矩阵形状从(3,H,W)变成(H,W,3)
        start = time.perf_counter()
        # res_img = res_img.to('cpu').numpy() * 255.
        print('Copy use:', time.perf_counter() - start)

        r = Image.fromarray(res_img[0]).convert('L')
        g = Image.fromarray(res_img[1]).convert('L')
        b = Image.fromarray(res_img[2]).convert('L')
        # print(res_img)
        print("Output shape:", res_img.shape)
        # size = shape[1], shape[0]
        # res_img = Image.fromarray(res * 255.,mode='RGB')
        # PIL.Image.merge(mode, bands)
        res_img = Image.merge('RGB', (r, g, b))

        return res_img.convert('RGB')

    def run(self, raw_img):
        blks, h, w = self.padding_and_img_to_blks(raw_img)
        return self.run_forward(blks=blks, h=h, w=w)

    def test(self, save_path="./test_result/", filename="test_result_PSNR", test_list=range(0, 10)):
        # random_pick = random.sample(range(0, 100), 100)
        random_pick = range(0, 10)
        PSNR_SUM = 0
        SSIM_SUM = 0
        for i in test_list:
            data_path, label_path = self.imgs[i]
            data = Image.open(data_path).convert('L')
            label = Image.open(label_path).convert('RGB')
            # data.show()
            # label.show()
            Net_img = self.run(data)
            # Net_img.show()

            label_np, Net_img_np = self.to_same_size_ndarray(label, Net_img)
            PSNR = compare_psnr(im_true=label_np, im_test=Net_img_np)
            # http://www.voidcn.com/article/p-auyocqzg-bac.html
            SSIM = compare_ssim(X=label_np, Y=Net_img_np, win_size=11, multichannel=True)
            print("PSNR:", PSNR, "SSIM:", SSIM)

            PSNR_SUM += PSNR
            SSIM_SUM += SSIM

            PSNR = str(PSNR)
            SSIM = str(SSIM)

            str_write = ''
            str_write += ("No." + str(i + 1) + ": " + PSNR + "\n")
            str_write += ("No." + str(i + 1) + ": " + SSIM + "\n")
            self.save_res(str_write, save_path, filename)

            # random_ID = str(random.randint(0, 100000000))
            Net_img.save(save_path + filename + str(i + 1) + '_PSNR=' + PSNR[:7] + ".TIF", "TIFF")

            label.save(save_path + filename + str(i + 1) + "_Real" + ".TIF", "TIFF")

        avg_str = ""
        avg_str += "PSNR_AVG=" + str(PSNR_SUM / len(test_list))[:7] + "\n"
        avg_str += "SSIM_AVG=" + str(SSIM_SUM / len(test_list))[:7] + "\n"
        print("AVG: ", avg_str)
        self.save_res(avg_str, save_path, filename)

    def save_res(self, contents, save_path, filename):
        filename = save_path + filename + '.txt'
        fh = open(filename, 'a')
        fh.write(contents)
        fh.close()

    def PSNR(self, A, B):
        A = np.array(A)
        B = np.array(B)
        h = min(A.shape[0], B.shape[0])
        w = min(A.shape[1], B.shape[1])
        A = A[:h, :w]
        B = B[:h, :w]
        mse = ((A.astype(np.float) - B.astype(np.float)) ** 2).mean()
        if mse == 0: return 10e4
        print("MSE: ", mse)
        return 10 * np.log10((255.0 ** 2) / mse)

    def to_same_size_ndarray(self, A: Image, B: Image) -> (np.ndarray, np.ndarray):
        A = np.array(A)
        B = np.array(B)
        h = min(A.shape[0], B.shape[0])
        w = min(A.shape[1], B.shape[1])
        A = A[:h, :w]
        B = B[:h, :w]
        # A = np.swapaxes(A, 1, 2)
        # A = np.swapaxes(A, 0, 1)
        # B = np.swapaxes(B, 1, 2)
        # B = np.swapaxes(B, 0, 1)
        return A, B


def Run_test(net, model_path, test_data_path, save_to, as_name, _block_size=32, _test_list=range(0, 10)):
    torch.set_default_tensor_type('torch.DoubleTensor')
    print("Creating the model...")
    MyNet = net.double().to(device)
    print("Loading the model data...")
    MyNet.load_state_dict(torch.load(model_path, map_location=device))
    print("Init...")
    Mytest = Test(test_data_path, MyNet, block_size=_block_size)
    Mytest.test(save_path=save_to, filename=as_name, test_list=_test_list)


def To_Bayer(img):
    w, h = img.size
    # img=img.resize((int(w/2),int(h/2)), Image.ANTIALIAS)
    # w,h=img.size
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

    return bayer

    # Bayer_mono=Image.fromarray(bayer_mono)
    # Bayer_mono.convert('L')
    # Bayer_mono.convert('RGB')
    # Bayer_mono.show()
    # return Bayer_mono


def LR_to_HR(net, model_path, img_path, save_to, _block_size=32):
    torch.set_default_tensor_type('torch.DoubleTensor')
    print("Creating the model...")
    MyNet = net.double()
    print("Loading the model data...")
    MyNet.load_state_dict(torch.load(model_path))
    print("Init...")
    Mytest = Test(Net=MyNet, file_path="", block_size=_block_size)
    print("Read image...")
    img = Image.open(img_path).convert("RGB")
    img = To_Bayer(img)
    img = Mytest.run(raw_img=img.convert('L'))
    img.save(save_to + ".TIF", "TIFF")
