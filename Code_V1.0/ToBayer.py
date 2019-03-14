from PIL import Image
import numpy as np

def DownSize(img):
    for i in range(3):
        w,h=img.size
        print(int(w/1.25),int(h/1.25))
        img=img.resize((int(w/1.25),int(h/1.25)), Image.ANTIALIAS)
    return img

def To_Bayer(img):
    w,h=img.size
    img=img.resize((int(w/2),int(h/2)), Image.ANTIALIAS)
    w,h=img.size
    # r,g,b=img.split()
    data=np.array(img)
    """
    R G R G
    G B G B
    R G R G
    G B G B
    """
    bayer_mono=np.zeros((h,w))
    for r in range(h):
        for c in range(w):
            if(0==r%2):
                if(1==c%2):
                    data[r,c,0]=0
                    data[r,c,2]=0

                    bayer_mono[r,c]=data[r,c,1]
                else:
                    data[r,c,1]=0
                    data[r,c,2]=0

                    bayer_mono[r,c]=data[r,c,0]
            else:
                if(0==c%2):
                    data[r,c,0]=0
                    data[r,c,2]=0

                    bayer_mono[r,c]=data[r,c,1]
                else:
                    data[r,c,0]=0
                    data[r,c,1]=0

                    bayer_mono[r,c]=data[r,c,2]

    # 三通道Bayer图像
    bayer=Image.fromarray(data)
    #bayer.show()

    return bayer


dir="/Users/linweichen/Desktop/计算机学习资料/TrainData/RAISE_1K/Rename"

for i in range(1001,1001):
    img=Image.open(dir+"/raw"+str(i)+".TIF","r")
    # img.show()
    img=DownSize(img)
    # img.show()
    img.save("/Users/linweichen/Desktop/计算机学习资料/TrainData/RAISE_1K/Resize/real"+str(i)+".TIF","TIFF")
    # r,g,b=img.split()

    #print (type(h))
    # emtpy=Image.new("L",(w,h))
    # print(type(r))
    # r_merged = Image.merge('RGB',(r,emtpy,emtpy))
    # g_merged = Image.merge('RGB',(emtpy,g,emtpy))
    # b_merged = Image.merge('RGB',(emtpy,emtpy,b))
    # r_merged.show()
    # b_merged.show()
    # g_merged.show()

resize_dir="/Users/linweichen/Desktop/计算机学习资料/TrainData/RAISE_1K/Resize/real"
for i in range(1,1001):
    img=Image.open(resize_dir+str(i)+".TIF","r")
    # img.show()
    img=To_Bayer(img)
    img.save("/Users/linweichen/Desktop/计算机学习资料/TrainData/RAISE_1K/Bayer/Bayer"+str(i)+".TIF","TIFF")


# Python PIL image split to RGB
''' 
from PIL import Image

img = Image.open('ra.jpg')
data = img.getdata()

# Suppress specific bands (e.g. (255, 120, 65) -> (0, 120, 0) for g)
r = [(d[0], 0, 0) for d in data]
g = [(0, d[1], 0) for d in data]
b = [(0, 0, d[2]) for d in data]

img.putdata(r)
img.save('r.png')
img.putdata(g)
img.save('g.png')
img.putdata(b)
img.save('b.png')
'''
