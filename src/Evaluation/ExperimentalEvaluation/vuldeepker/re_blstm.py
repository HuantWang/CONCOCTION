from __future__ import print_function

import warnings

from keras.utils import to_categorical
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight

warnings.filterwarnings("ignore")

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, LeakyReLU
from keras.optimizers import Adamax
from keras.models import load_model

import random
import pandas as pd
from sklearn.model_selection import train_test_split
import os

"""
Bidirectional LSTM neural network
Structure consists of two hidden layers and a BLSTM layer
Parameters, as from the VulDeePecker paper:
    Nodes: 300
    Dropout: 0.5
    Optimizer: Adamax
    Batch size: 64
    Epochs: 4
"""
class BLSTM:
    def __init__(self, data, seed=0, name="", batch_size=64):
        vectors = np.stack(data.iloc[:, 0].values)
        labels = data.iloc[:, 1].values
        positive_idxs = np.where(labels == 1)[0]
        negative_idxs = np.where(labels == 0)[0]
        #是为了保证正负样本的数量一致，在初始化的时候保证一样就可以
        # undersampled_negative_idxs = np.random.choice(negative_idxs, len(positive_idxs), replace=True)#从样本中随机选择size大小的元素
        undersampled_negative_idxs = negative_idxs[0:len(positive_idxs)]
        resampled_idxs = np.concatenate([positive_idxs, undersampled_negative_idxs])

        #random_state的取值范围为0 - 2 ^ 32
        self.randomstate = seed
        # X_train, X_test, y_train, y_test = train_test_split(vectors[resampled_idxs, ], labels[resampled_idxs],
        #                                                     test_size=0.2, stratify=labels[resampled_idxs],random_state=self.randomstate)
        X_train, X_test, y_train, y_test = train_test_split(vectors[resampled_idxs,], labels[resampled_idxs],
                                                            test_size=0.4, stratify=labels[resampled_idxs],
                                                            random_state=seed)

        # print("==============random_state===========\n" ,self.randomstate)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)
        self.name = name
        self.batch_size = batch_size
        self.class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=labels)
        model = Sequential()
        model.add(Bidirectional(LSTM(300), input_shape=(vectors.shape[1], vectors.shape[2])))
        model.add(Dense(300))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(Dense(300))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        # Lower learning rate to prevent divergence
        adamax = Adamax(lr=0.002)
        model.compile(adamax, 'categorical_crossentropy', metrics=['accuracy'])
        self.model = model

        model.summary()
        print()

    """
    Trains model based on training data
    """


    def train(self):
        #修改此处epoch
        #方法一
        from keras import callbacks
        # remote = callbacks.RemoteMonitor(root='http://localhost:9000')
        #方法二
        # import keras
        # TensorBoardcallback = keras.callbacks.TensorBoard(log_dir='./logs/log6', histogram_freq=0, write_graph=True, write_grads=False,
        #                                                   write_images=False, embeddings_freq=0, embeddings_layer_names=None,
        #                                                   embeddings_metadata=None, embeddings_data=None, update_freq=16)
        #方法三
        from History import LossHistory
        history = LossHistory()

        self.model.fit(self.X_train, self.y_train,validation_split=0.2,batch_size=self.batch_size,
                       epochs=4,verbose=1,
                       class_weight=self.class_weight,callbacks=[history])

        t = history.time
        acc = history.acc

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(time, acc)
        # plt.savefig("easyplot.png")
        # plt.show()

        t = np.array(t)
        acc = np.array(acc)
        t = t.reshape(t.size, 1)
        acc = acc.reshape(acc.size, 1)
        data = np.concatenate((t, acc),axis=1)
        # 写入excel
        import pandas as pd
        import time
        # import datetime
        # nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        data = pd.DataFrame(data)

        filename = str(self.name+"_acc.xlsx")
        writer = pd.ExcelWriter(filename)  # 写入Excel文件
        data.to_excel(writer, index=False,header=["time","acc"],encoding="utf8")  # ‘page_1’是写入excel的sheet名
        writer.save()
        writer.close()

        self.model.save_weights(self.name + "_model.h5")
        # self.model.save(self.name + "_whole_model.h5")

    """
    Tests accuracy of model based on test data
    Loads weights from file if no weights are attached to model object
    """
    def test(self,modelname):
        # self.model.load_weights(self.name + "_model.h5")
        self.model.load_weights(modelname)
        # self.model.load_weights("sard_and_github1_model.h5")
        values = self.model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size)
        print("Accuracy is...", values[1])
        predictions = (self.model.predict(self.X_test, batch_size=self.batch_size)).round()

        tn, fp, fn, tp = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(predictions, axis=1)).ravel()
        # print('False positive rate is...', fp / (fp + tn))
        # print('False negative rate is...', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('True recall is...', recall)
        precision = tp / (tp + fp)
        print('Precision is...', precision)
        f1 = (2 * precision * recall) / (precision + recall)
        print('F1 score is...',f1 )

        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        tnr = tn/(fp+tn)
        fnr = fn/(tp+fn)

        #写人excel
        data = pd.DataFrame([[os.path.basename(self.name),os.path.basename(modelname),self.randomstate,values[1],recall,precision,f1,tpr,fpr,tnr,fnr]])

        filename = str(self.name+"_result.xlsx")
        writer = pd.ExcelWriter(filename)  # 写入Excel文件
        data.to_excel(writer, index=False,header=["name","rename","seed","acc","recall","precision","f1","tpr","fpr","tnr","fnr"],encoding="utf8")  # ‘page_1’是写入excel的sheet名
        writer.save()
        writer.close()