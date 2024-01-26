# -*- coding: utf-8 -*-

import sys
import os
import time

import keras
import numpy as np
import pandas as pd
from keras import Model

from keras import initializers
from keras import layers
from keras.initializers import glorot_uniform

from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Activation, Input, ZeroPadding2D, AveragePooling2D
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.models import Sequential, load_model
from keras.optimizers import SGD,Adam
from sklearn.metrics import confusion_matrix
# from tensorflow import set_random_seed
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc 
import tensorflow as tf


import keras.backend as K
K.set_image_data_format("channels_last")
K.set_learning_phase(1)

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

np.random.seed(11)
tf.set_random_seed(12)


class Metric(object):       
    def __init__(self, y_true, y_pred):
        self.__matrix = confusion_matrix(y_true, y_pred)   

    def Matrix(self):
        return self.__matrix

    def TP(self):
        tp = np.diag(self.__matrix)
        return tp.astype(float)

    def TN(self):
        tn = self.__matrix.sum() - (self.FP() + self.FN() + self.TP())
        return tn.astype(float)

    def FP(self):
        fp = self.__matrix.sum(axis=0) - np.diag(self.__matrix)
        return fp.astype(float)

    def FN(self):
        fn = self.__matrix.sum(axis=1) - np.diag(self.__matrix)
        return fn.astype(float)

    def TPRate(self):      
        return self.TP() / (self.TP() + self.FN() + sys.float_info.epsilon)  

    def TNRate(self):
        return self.TN() / (self.TN() + self.FP() + sys.float_info.epsilon)

    def FPRate(self):
        return 1 - self.TNRate()

    def FNRate(self):
        return 1 - self.TPRate()

    def Accuracy(self):
        ALL = self.TP() + self.FP() + self.TN() + self.FN()
        RIGHT = self.TP() + self.TN()
        return RIGHT / (ALL + sys.float_info.epsilon)

    def Recall(self):
        return self.TP() / (self.TP() + self.FN() + sys.float_info.epsilon)

    def Precision(self):
        return self.TP() / (self.TP() + self.FP() + sys.float_info.epsilon)

    def TSS(self):
        return self.TPRate() - self.FPRate()

    def HSS(self):
        P = self.TP() + self.FN()
        N = self.TN() + self.FP()
        up = 2 * (self.TP() * self.TN() - self.FN() * self.FP())
        below = P * (self.FN() + self.TN()) + N * (self.TP() + self.FP())
        return up / (below + sys.float_info.epsilon)


class Read(object):
    def __init__(self, directory, order):
        def read_csv(name):
            return pd.read_csv(name, header=None, delimiter=",").values

        names = [directory + "/" + str(order) + "_positive_test.csv",
                  directory + "/" + str(order) + "_positive_train.csv",
                  directory + "/" + str(order) + "_negative_test.csv",
                  directory + "/" + str(order) + "_negative_train.csv"]   

        self.__positive_test, self.__positive_train, \
        self.__negative_test, self.__negative_train = map(read_csv, names)  
                                                      
        
    def get_original_data(self):
        return self.__positive_test, self.__positive_train, \
               self.__negative_test, self.__negative_train



class Parse(object):
    def __init__(self, _positive_test, _positive_train, _negative_test, _negative_train, is_vgg):
        # is_vgg flag
        self.__is_vgg = is_vgg
        # return data
        self.__train_x, self.__train_y, self.__test_x, self.__test_y = [], [], [], []
        # class_weight data
        # [image_number, [AR]]
        self.__zero, self.__one = [0, []], [0, []]
        for line in _positive_test:
            if line[1] == 1:
                if self.__is_vgg is False:
                    self.__test_x.append(line[4:])     
                else:
                    self.__test_x.append([line[4:], line[4:], line[4:]])
                self.__test_y.append(line[3])
        for line in _negative_test:
            if line[1] == 1:
                if self.__is_vgg is False:
                    self.__test_x.append(line[4:])
                else:
                    self.__test_x.append([line[4:], line[4:], line[4:]])
                self.__test_y.append(line[3])
        for line in _positive_train:
            if self.__is_vgg is False:
                self.__train_x.append(line[4:])
            else:
                self.__train_x.append([line[4:], line[4:], line[4:]])      
            self.__train_y.append([line[2], line[3]])        
        for line in _negative_train:
            if self.__is_vgg is False:
                self.__train_x.append(line[4:])
            else:
                self.__train_x.append([line[4:], line[4:], line[4:]])
            self.__train_y.append([line[2], line[3]])

    def __prepare_data(self):
        level_map = {"N": 0., "C": 1., "M": 1., "X": 1.}    
        for indices in range(len(self.__train_y)):
            if level_map[self.__train_y[indices][1]] == 0:    
                self.__zero[0] += 1                           
                self.__zero[1].append(self.__train_y[indices][0])  
            elif level_map[self.__train_y[indices][1]] == 1:  
                self.__one[0] += 1          
                self.__one[1].append(self.__train_y[indices][0])
            self.__train_y[indices] = level_map[self.__train_y[indices][1]]  
        for indices in range(len(self.__test_y)):
            self.__test_y[indices] = level_map[self.__test_y[indices]]    

    def class_weight(self, zero=True, alpha=1., beta=1.):
        if zero:
            result = self.__zero[0] / 10000 * len(list(set(self.__zero[1]))) / 100 * alpha  
            print("Zero percent:", result) 
            return result
        else:
            result = self.__one[0] / 10000 * len(list(set(self.__one[1]))) / 100 * beta   
            print("One percent:", result)
            return result

    def load_data(self):
        # prepare data
        self.__prepare_data()
        if self.__is_vgg is False:
            # process train_X
            self.__train_x = np.array(self.__train_x, dtype=np.float32)
            self.__train_x /= 4000     
            self.__train_x = self.__train_x.reshape((self.__train_x.shape[0], 128, 128, 1))  
            # process test_x
            self.__test_x = np.array(self.__test_x, dtype=np.float32)
            self.__test_x /= 4000
            self.__test_x = self.__test_x.reshape((self.__test_x.shape[0], 128, 128, 1)) 
        self.__train_y = keras.utils.to_categorical(self.__train_y, int(max(self.__train_y) + 1)) 
        self.__test_y = keras.utils.to_categorical(self.__test_y, int(max(self.__test_y) + 1)) 
        return self.__train_x, self.__train_y, self.__test_x, self.__test_y  

def identity_block(X,f,filters,stage,block):
    """
    三层的恒等残差块
    param :
    X -- 输入的张量，维度为（m, n_H_prev, n_W_prev, n_C_prev）
    f -- 整数，指定主路径的中间 CONV 窗口的形状
    filters -- python整数列表，定义主路径的CONV层中的过滤器数目
    stage -- 整数，用于命名层，取决于它们在网络中的位置
    block --字符串/字符，用于命名层，取决于它们在网络中的位置
    return:
    X -- 三层的恒等残差块的输出，维度为：(n_H, n_W, n_C)
    """
    
    conv_name_base = "res"+str(stage)+block+"_branch"
    bn_name_base = "bn"+str(stage)+block+"_branch"
 
    F1,F2,F3 = filters
    
    X_shortcut = X
 
    #主路径第一部分
    X = Conv2D(filters=F1,kernel_size=(1,1),strides=(1,1),padding="valid",
               name=conv_name_base+"2a",kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2a")(X)   ## 3:在通道/厚度轴上标准化
    X = Activation("relu")(X)
 
    # 主路径第二部分
    X = Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),padding="same",
               name=conv_name_base+"2b",kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2b")(X)
    X = Activation("relu")(X)
 
    # 主路径第三部分
    X = Conv2D(filters=F3,kernel_size=(1,1),strides=(1,1),padding="valid",
               name=conv_name_base+"2c",kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name=bn_name_base+"2c")(X)
 
    # 主路径最后部分,为主路径添加shortcut并通过relu激活
    X = layers.add([X,X_shortcut])
    X = Activation("relu")(X)
 
    return X
 
tf.reset_default_graph()
with tf.Session() as sess:
    np.random.seed(1)
    A_prev = tf.placeholder("float",shape=[3,4,4,6])
    X = np.random.randn(3,4,4,6)
    A = identity_block(A_prev,f=2,filters=[2,4,6],stage=1,block="a")
    sess.run(tf.global_variables_initializer())
    out = sess.run([A],feed_dict={A_prev:X,K.learning_phase():0})
    print("out = "+str(out[0][1][1][0]))
 
#卷积残差块——convolutional_block
def convolutional_block(X,f,filters,stage,block,s=2):
    """
    param :
    X -- 输入的张量，维度为（m, n_H_prev, n_W_prev, n_C_prev）
    f -- 整数，指定主路径的中间 CONV 窗口的形状（过滤器大小，ResNet中f=3）
    filters -- python整数列表，定义主路径的CONV层中过滤器的数目
    stage -- 整数，用于命名层，取决于它们在网络中的位置
    block --字符串/字符，用于命名层，取决于它们在网络中的位置
    s -- 整数，指定使用的步幅
    return:
    X -- 卷积残差块的输出，维度为：(n_H, n_W, n_C)
    """   
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"
 
    F1, F2, F3 = filters
 
    X_shortcut = X
 
    # 主路径第一部分
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding="valid",
               name=conv_name_base + "2a", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)
 
    # 主路径第二部分
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",
               name=conv_name_base + "2b", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)
 
    # 主路径第三部分
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name=conv_name_base + "2c", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)
 
    #shortcut路径
    X_shortcut = Conv2D(filters=F3,kernel_size=(1,1),strides=(s,s),padding="valid",
               name=conv_name_base+"1",kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3,name=bn_name_base+"1")(X_shortcut)
 
    # 主路径最后部分,为主路径添加shortcut并通过relu激活
    X = layers.add([X, X_shortcut])
    X = Activation("relu")(X)
 
    return X
 
tf.reset_default_graph()
 
with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = convolutional_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
    test.run(tf.global_variables_initializer())
    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
    print("out = " + str(out[0][1][1][0]))

#50层ResNet模型构建
def ResNet50(input_shape = (128,128,1),classes = 2):
    """
    构建50层的ResNet,结构为：    这4层整体为1       3            6            3             9
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
           3             15           3          6               1               = 50
    param :                                    
    input_shape -- 数据集图片的维度
    classes -- 整数，分类的数目
    return:
    model -- Keras中的模型实例
    """
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3,3))(X_input) 
 
    # Stage 1
    X = Conv2D(64,kernel_size=(7,7),strides=(2,2),name="conv1",kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3,name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)
 
    # Stage 2
    X = convolutional_block(X,f=3,filters=[64,64,256],stage=2,block="a",s=1)
    X = identity_block(X,f=3,filters=[64,64,256],stage=2,block="b")
    X = identity_block(X,f=3,filters=[64,64,256],stage=2,block="c")
 
    #Stage 3
    X = convolutional_block(X,f=3,filters=[128,128,512],stage=3,block="a",s=2)
    X = identity_block(X,f=3,filters=[128,128,512],stage=3,block="b")
    X = identity_block(X,f=3,filters=[128,128,512],stage=3,block="c")
    X = identity_block(X,f=3,filters=[128,128,512],stage=3,block="d")
 
    # Stage 4
    X = convolutional_block(X,f=3,filters=[256,256,1024],stage=4,block="a",s=2)
    X = identity_block(X,f=3,filters=[256,256,1024],stage=4,block="b")
    X = identity_block(X,f=3,filters=[256,256,1024],stage=4,block="c")
    X = identity_block(X,f=3,filters=[256,256,1024],stage=4,block="d")
    X = identity_block(X,f=3,filters=[256,256,1024],stage=4,block="e")
    X = identity_block(X,f=3,filters=[256,256,1024],stage=4,block="f")
 
    #Stage 5
    X = convolutional_block(X,f=3,filters=[512,512,2048],stage=5,block="a",s=2)
    X = identity_block(X,f=3,filters=[256,256,2048],stage=5,block="b")
    X = identity_block(X,f=3,filters=[256,256,2048],stage=5,block="c")
 
    X = AveragePooling2D(pool_size=(2,2))(X)
 
    
    X = Flatten()(X)   
    X = Dense(classes,activation="softmax",name="fc"+str(classes),kernel_initializer=glorot_uniform(seed=0))(X)  
 
    
    model = Model(inputs=X_input,outputs=X,name="ResNet50")
    
 
    return model


class TrainModel(object):   
    def __init__(self, conf):
        self.__conf = conf
        

    def train(self, move):
        reader = Read("./data", move)
        
        a, b, c, d = reader.get_original_data()    
        
        parser = Parse(a, b, c, d, self.__conf["is_vgg"])
        train_x, train_y, test_x, test_y = parser.load_data()

        if not self.__conf["is_vgg"]:
            # train normal
            model = ResNet50(input_shape=(128, 128, 1),classes = 2)
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=SGD(lr=self.__configure_maps["1"]["learning_rate"],  
                                    momentum=self.__configure_maps["1"]["momentum"],      
                                    decay=self.__configure_maps["1"]["decay"],            
                                    nesterov=self.__configure_maps["nesterov"]),          
                          metrics=["accuracy"])

            model.fit(train_x, train_y, batch_size=self.__conf["1"]["batch_size"],   
                      epochs=self.__conf["1"]["epochs"], verbose=self.__conf["verbose"],     
                      class_weight={0.: parser.class_weight(alpha=self.__conf["1"]["alpha"]),
                                    1.: parser.class_weight(False, beta=self.__conf["1"]["beta"])},   
                      validation_data=(test_x, test_y),                                 
                      callbacks=[ModelCheckpoint("./model/2classification/" + str(move) + ".h5",
                                                 monitor='val_loss', verbose=1, mode='min',
                                                 save_best_only=True, save_weights_only=False)])   
            model = load_model("./model/2classification/" + str(move) + ".h5")
            y_true, y_pred = test_y, model.predict(test_x) 
            return y_true.argmax(axis=1), y_pred.argmax(axis=1)    


def Main(is_vgg):
    all_nums = 7  
    std_file_name = "./logs/" + time.strftime("%Y_%b_%d_%H_%M_%S", time.gmtime()) + ".txt"  
    
    sys.stdout = open(std_file_name, "w+")
    conf = {"num_of_classes": 2, "nesterov": True, "verbose": 2, "is_vgg": is_vgg,
            "1": {
                "batch_size": 16, "epochs": 80, "learning_rate": 0.0001,          
                "momentum": 0.5, "decay": 5 * 1e-6, "alpha": 1, "beta": 0.8,
                "input_shape": (128, 128, 1),
            }
            # ,"2": {
            #     "batch_size": 16, "epochs": 80, "learning_rate": 4 * 1e-4,
            #     "momentum": 0.6, "decay": 1e-6, "alpha": 1., "beta": 0.8,
            #     "input_shape": (128, 128, 3),
            # }
            }
    all_metric = {"Recall": [0, 0], "Precision": [0, 0], "Accuracy": [0, 0],
                  "TSS": [0, 0], "HSS": [0, 0]}
    all_matrix = np.array([[0, 0], [0, 0]])
    for i in range(all_nums):
        trainer = TrainModel(conf)
        a, b = trainer.train(i)   
        
        fpr,tpr,threshold = roc_curve(a, b) 
        roc_auc = auc(fpr,tpr)                  
        plt.figure()
        lw = 2
        plt.figure(figsize=(10,10))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) 
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig(str(i) + 'ROC.png')
                          
        metric = Metric(a, b)
        print("Matrix:\n", metric.Matrix())
        print("Recall:", metric.Recall())
        print("Precision:", metric.Precision())
        print("Accuracy:", metric.Accuracy())
        print("TSS:", metric.TSS())
        print("HSS:", metric.HSS())
        sys.stdout.flush()

        all_matrix += metric.Matrix()
        all_metric["Recall"] += metric.Recall()
        all_metric["Precision"] += metric.Precision()
        all_metric["Accuracy"] += metric.Accuracy()
        all_metric["TSS"] += metric.TSS()
        all_metric["HSS"] += metric.HSS()

    print("\n-----------------------")
    print(all_matrix)
    for index in all_metric:
        print(index, np.array(all_metric[index]) / all_nums)


if __name__ == '__main__':
    Main(False)                          
