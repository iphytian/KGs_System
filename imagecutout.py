#被RF判定为1的图片进行图片切割及图像预处理
# Image cutting and image preprocessing for pictures judged as 1 by RF

from astropy.io import fits
import numpy as np
import sep
from PIL import Image

def get_max_value(martix):
    res_list = []
    for j in range(len(martix[0])):
        one_list = []
        for i in range(len(martix)):
            one_list.append(int(martix[i][j]))
        res_list.append(str(max(one_list)))
    maxlist=max(res_list)
    return maxlist

def cutoutimage(image_path):
    cnn_set=[]
    data_0 = fits.open(image_path)
    data_0 = data_0[0].data
    data_1 = data_0[133:992,133:992]

    # 检测图像背景，获得去除背景后的图像
    # Detect the image background and obtain the image after removing the background
    m, s = np.mean(data_1), np.std(data_1)
    data_1 = data_1.astype(np.float64)
    bkg = sep.Background(data_1, mask=None, bw=64, bh=64, fw=3, fh=3)
    data_sub = data_1 - bkg

    # 提取出前两颗亮星的64*64的图像
    # Extract the 64*64 images of the first two bright stars
    objects = sep.extract(data_sub, 2.5, err=bkg.globalrms, deblend_nthresh=1)
    npix=[]
    idex1=[]
    c=[]
    n = 0
    number = 0
    for i in range(len(objects)):
        a = objects[i][15]
        b = objects[i][16]
        a = max(a, b)
        b = min(a, b)
        if a < 32 and b > 2.5:
            number = number + 1
            c.append(objects[i])
        else:
            number = number

    for a in range(len(c)):
        npix.append(c[a][2])
    # 判断亮星数目是否大于2
    # Determine whether the number of bright stars is greater than 2
    if number >= 2:
        idex1.append(npix.index(max(npix)))
        npix[idex1[0]]=0
        idex1.append(npix.index(max(npix)))
    elif number == 1:
        idex1.append(npix.index(max(npix)))
    else:
        num = 0

    if len(idex1):
        for i in idex1:
            a1 = int(c[i][7])+132
            a2 = int(c[i][8])+132
            image = data_0[a2-32:a2+32,a1-32:a1+32]
            # 进行图像预处理
            # Perform image preprocessing
            amax = get_max_value(image)
            if int(amax) > 0:
                b = 255/int(amax)*image
            else:
                b = image
            b = b - np.mean(b, keepdims=True)
            b = Image.fromarray(b)
            b = b.convert('L')
            newpath = '.\\dataset_cnn\\' + str(n) + '.png'
            b.save(newpath)
            n=n+1
    # 返回亮星数目
    # Return the number of bright stars
    num = len(idex1)
    return num