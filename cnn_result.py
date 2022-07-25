# 给出CNN分类结果
# Second Classes Session

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
import os
import sklearn
from PIL import Image

def generateds(txt):
    f = open(txt, 'r')
    contents = f.readlines()  # 读取文件中所有行 Read all lines in the file
    f.close()
    x, y_ = [], []
    for content in contents:
        # 以空格分开，图片路径为value[0] , 标签为value[1] , 存入列表
        '''Separate by spaces, the image path is value[0], 
        the label is value[1], and it is stored in the list '''
        value = content.split()
        img_path = value[0]
        img = Image.open(img_path)
        img = np.array(img.convert('L'))
        img = img / 255.
        x.append(img)
        y_.append(value[1])
        print('loading : ' + content)
    x = np.array(x)
    y_ = np.array(y_)
    y_ = y_.astype(np.int64)
    return x, y_  # 返回输入特征x，返回标签y_ Return input feature x, return label y_

train_txt = '.\\dataset_cnn\\iphy123.txt'
x_train_savepath = '.\\dataset_cnn\\x_train.npy'
y_train_savepath = '.\\dataset_cnn\\y_train.npy'

if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath):
    print('-------------Load Datasets-----------------')
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_train = np.reshape(x_train_save, (len(x_train_save), 64, 64))
else:
    print(1)
    print('-------------Generate Datasets-----------------')
    x_train, y_train = generateds(train_txt)

    print('--------------Save Datasets-----------------')
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    np.save(x_train_savepath, x_train_save)
    np.save(y_train_savepath, y_train)


np.set_printoptions(threshold=np.inf)

# 数据集导入
# Data set import
x_train,  x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_train, y_train, random_state=10, train_size=0.6,test_size=0.4)

# 构成4维数组
# Form a 4-dimensional array
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

###############################################    Split Line   ###############################################
# 构建用于训练的卷积函数
# construct a convolution function for training
def CNN_train():
    class Baseline(Model):
        def __init__(self):
            super(Baseline, self).__init__()
            self.c1 = Conv2D(filters=32, kernel_size=(8, 8), padding='valid')  # 卷积层
            self.b1 = BatchNormalization()
            self.a1 = Activation('relu')
            self.p1 = MaxPool2D(pool_size=(7, 7), strides=2, padding='valid')  # 池化层
            self.d1 = Dropout(0.2)

            self.c2 = Conv2D(filters=64, kernel_size=(6, 6), padding='valid')  # 卷积层
            self.b2 = BatchNormalization()
            self.a2 = Activation('relu')
            self.p2 = MaxPool2D(pool_size=(5, 5), strides=2, padding='valid')  # 池化层
            self.d2 = Dropout(0.2)

            self.c3 = Conv2D(filters=128, kernel_size=(4, 4), padding='valid')  # 卷积层
            self.b3 = BatchNormalization()
            self.a3 = Activation('relu')
            self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')  # 池化层
            self.d3 = Dropout(0.2)

            self.flatten = Flatten()
            self.f1 = Dense(64, activation='relu')
            self.d1 = Dropout(0.2)
            self.f2 = Dense(64, activation='relu')
            self.d2 = Dropout(0.2)
            self.f3 = Dense(6, activation='softmax')

        def call(self, x):
            x = self.c1(x)
            x = self.b1(x)
            x = self.a1(x)
            x = self.p1(x)
            x = self.d1(x)

            x = self.c2(x)
            x = self.b2(x)
            x = self.a2(x)
            x = self.p2(x)
            x = self.d2(x)

            x = self.c3(x)
            x = self.b3(x)
            x = self.a3(x)
            x = self.p3(x)
            x = self.d3(x)

            x = self.flatten(x)
            x = self.f1(x)
            x = self.d1(x)
            x = self.f2(x)
            x = self.d2(x)
            y = self.f3(x)
            return y

    model = Baseline()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    # 保存训练好的参数
    # Save the trained parameters
    checkpoint_save_path = "./checkpoint/Baseline.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)

    history = model.fit(x_train, y_train, batch_size=100, epochs=1, validation_data=(x_test, y_test), validation_freq=1,
                        callbacks=[cp_callback])
    model.summary()

    file = open('./weights.txt', 'w')
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy()) + '\n')
    file.close()

###############################################    Split Line   ###############################################

def CNN_clf(path):
    # 导入训练好的cnn结构
    # Import the trained CNN structure
    model_save_path = './checkpoint/Baseline.ckpt'

    class Baseline(Model):
        def __init__(self):
            super(Baseline, self).__init__()
            self.c1 = Conv2D(filters=32, kernel_size=(8, 8), padding='valid')  # 卷积层
            self.b1 = BatchNormalization()
            self.a1 = Activation('relu')
            self.p1 = MaxPool2D(pool_size=(7, 7), strides=2, padding='valid')  # 池化层
            self.d1 = Dropout(0.2)

            self.c2 = Conv2D(filters=64, kernel_size=(6, 6), padding='valid')  # 卷积层
            self.b2 = BatchNormalization()
            self.a2 = Activation('relu')
            self.p2 = MaxPool2D(pool_size=(5, 5), strides=2, padding='valid')  # 池化层
            self.d2 = Dropout(0.2)

            self.c3 = Conv2D(filters=128, kernel_size=(4, 4), padding='valid')  # 卷积层
            self.b3 = BatchNormalization()
            self.a3 = Activation('relu')
            self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')  # 池化层
            self.d3 = Dropout(0.2)

            self.flatten = Flatten()
            self.f1 = Dense(64, activation='relu')
            self.d1 = Dropout(0.2)
            self.f2 = Dense(64, activation='relu')
            self.d2 = Dropout(0.2)
            self.f3 = Dense(6, activation='softmax')

        def call(self, x):
            x = self.c1(x)
            x = self.b1(x)
            x = self.a1(x)
            x = self.p1(x)
            x = self.d1(x)

            x = self.c2(x)
            x = self.b2(x)
            x = self.a2(x)
            x = self.p2(x)
            x = self.d2(x)

            x = self.c3(x)
            x = self.b3(x)
            x = self.a3(x)
            x = self.p3(x)
            x = self.d3(x)

            x = self.flatten(x)
            x = self.f1(x)
            x = self.d1(x)
            x = self.f2(x)
            x = self.d2(x)
            y = self.f3(x)
            return y

    model = Baseline()

    model.load_weights(model_save_path)

    # 图像预处理
    # Image preprocessing
    img = Image.open(path)
    img = np.array(img.convert('L'))
    img = img / 255.
    img = np.array(img)
    img = np.expand_dims(img, axis=2)

    x_predict = img[tf.newaxis, ...]
    result = model.predict(x_predict)
    pred = tf.argmax(result, axis=1)
    pred = int(pred)
    if pred == 6:
        result == [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    else:
        result = result
    print(result)
    print('\n')
    print(pred)
    return(result)  # 返回预测结果 Return prediction result