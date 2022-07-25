# 定义RF分类及预测所需要用到的函数
# Initial Classes Session

import numpy as np
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
from skimage import color, img_as_ubyte
import sep
import joblib
from astropy.io import fits
import sklearn
from sklearn.ensemble import RandomForestClassifier

# 灰度共生矩阵参数
# GLCM properties
def contrast_feature(matrix_coocurrence):
    contrast = greycoprops(matrix_coocurrence, 'contrast')
    return contrast

def dissimilarity_feature(matrix_coocurrence):
    dissimilarity = greycoprops(matrix_coocurrence, 'dissimilarity')
    return dissimilarity

def homogeneity_feature(matrix_coocurrence):
    homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
    return homogeneity

def asm_feature(matrix_coocurrence):
    asm = greycoprops(matrix_coocurrence, 'ASM')
    return asm

def energy_feature(matrix_coocurrence):
    energy = greycoprops(matrix_coocurrence, 'energy')
    return energy

def correlation_feature(matrix_coocurrence):
    correlation = greycoprops(matrix_coocurrence, 'correlation')
    return correlation

## 对RF进行训练
## Train RF
def svm_train():
    # 读入定标好的txt数据
    # Read in the calibrated .txt data
    path = '.\\test0806test.txt'
    f = open(path,'r')

    # 划分数据与标签
    # Divide data and labels
    x=[]
    y=[]
    contents = f.readlines()
    for content in contents:
        value = content.split()
        x.append(value[1:10])
        y.append(value[10])

    # 划分训练集和测试集
    # Divide training set and test set
    train_data,test_data,train_label,test_label =sklearn.model_selection.train_test_split(x,y, random_state=1, train_size=0.6,test_size=0.4)

    # 训练RF
    # train RF
    n_features = 9
    classifier = RandomForestClassifier(n_estimators=50, class_weight='balanced', max_features=None, max_depth=30, min_samples_split=2,
                                        bootstrap=True)
    classifier.fit(train_data, train_label)

    # 给出RF训练及测试精度
    # Give RF training and test accuracy
    print("训练集：",classifier.score(train_data,train_label))
    print("测试集：",classifier.score(test_data,test_label))

    # 保存训练好的RF
    # Save the trained RF
    joblib.dump(classifier, './rf_clf.m')

# 使用训练好的RF分类
# Use trained RF classification
def svm_clf(path):
    data_set = []
    data1 =fits.open(path)
    data = data1[0].data
    data = data[132:991, 132:991]

    # 检测图像背景，获得去除背景后的图像
    # Detect the background of the picture and obtain the picture after removing the background
    data = data.astype(np.float64)
    bkg = sep.Background(data, mask=None, bw=64, bh=64, fw=3, fh=3)
    data_sub = data - bkg  # 得到去噪后的数据
    objects = sep.extract(data_sub, 2.5, err=bkg.globalrms, deblend_nthresh=1)

    # 获得亮星数目
    # Get the number of bright stars
    number = 0
    for i in range(len(objects)):
        a = objects[i][15]
        b = objects[i][16]
        a = max(a, b)
        b = min(a, b)
        # 控制星象大小
        # Control star size
        if a < 32 and b > 2.5:
            number = number + 1
        else:
            number = number

    m1, s1 = np.mean(data_sub), np.std(data_sub)
    data_sub = data_sub.astype(np.uint16)

    # 获得灰度共生矩阵参数
    # Obtain gray level co-occurrence matrix parameters
    gray = color.rgb2gray(data_sub)
    image = img_as_ubyte(gray)
    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])  # 16-bit
    inds = np.digitize(image, bins)
    max_value = inds.max() + 1
    matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=max_value,
                                      normed=False, symmetric=False)
    cons = np.sum(contrast_feature(matrix_coocurrence)) / 4
    diss = np.sum(dissimilarity_feature(matrix_coocurrence)) / 4
    homo = np.sum(homogeneity_feature(matrix_coocurrence)) / 4
    asmm = np.sum(asm_feature(matrix_coocurrence)) / 4
    ener = np.sum(energy_feature(matrix_coocurrence)) / 4
    corr = np.sum(correlation_feature(matrix_coocurrence)) / 4
    # 熵的计算
    # Entropy calculation
    shan = shannon_entropy(image)
    data_set = [[m1,number,corr,s1,homo,shan,asmm,ener,cons]]

    # 加载保存好的模型进行预测
    # Load the saved model for prediction
    clf = joblib.load('./rf_clf.m')
    a=clf.predict(data_set)
    a=int(a[0])
    print(a)
    cnn_set=[path,m1,number,corr,s1,homo,shan,asmm,ener,cons,diss,a]
    return cnn_set