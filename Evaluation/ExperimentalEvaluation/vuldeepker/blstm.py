
from __future__ import print_function

import warnings

from keras.utils import to_categorical
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight
import os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt

import nni
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, LeakyReLU
from keras.optimizers import Adamax
from keras.models import load_model
import keras_metrics
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import datetime
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
    def __init__(self, data, args):
        vectors = np.stack(data.iloc[:, 0].values)
        labels = data.iloc[:, 1].values
        positive_idxs = np.where(labels == 1)[0]
        negative_idxs = np.where(labels == 0)[0]
        #是为了保证正负样本的数量一致，在初始化的时候保证一样就可以
        # undersampled_negative_idxs = np.random.choice(negative_idxs, len(positive_idxs), replace=True)#从样本中随机选择size大小的元素
        undersampled_negative_idxs = negative_idxs[0:len(positive_idxs)]
        resampled_idxs = np.concatenate([positive_idxs, undersampled_negative_idxs])

        #random_state的取值范围为0 - 2 ^ 32
        # self.randomstate = random.randint(1,pow(2,32))
        self.randomstate =42
        X_train, X_test, y_train, y_test = train_test_split(vectors[resampled_idxs, ], labels[resampled_idxs],
                                                            test_size=0.2, stratify=labels[resampled_idxs],random_state=self.randomstate)
        # X_train, X_test, y_train, y_test = train_test_split(vectors[resampled_idxs,], labels[resampled_idxs],
        #                                                     test_size=0.2, stratify=labels[resampled_idxs],
        #                                                     random_state=s)

        print("==============random_state===========\n" ,self.randomstate)
        self.X_train = X_train
        self.X_test = X_test
        self.f1=0
        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)
        self.batch_size = args.batch_size
        self.epochs=args.epochs
        self.class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=labels)
        model = Sequential()
        model.add(Bidirectional(LSTM(300), input_shape=(vectors.shape[1], vectors.shape[2])))
        model.add(Dense(300))
        model.add(LeakyReLU())
        model.add(Dropout(args.dropout))
        model.add(Dense(300))
        model.add(LeakyReLU())
        model.add(Dropout(args.dropout))
        model.add(Dense(2, activation=args.active))
        # Lower learning rate to prevent divergence
        adamax = Adamax(lr=args.lr)

        model.compile(adamax, 'categorical_crossentropy', metrics=[keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score(),
                                    'accuracy'])
        self.model = model
        model.summary()
        if args.mode=='pre':
            self.model_to_load=args.model_to_load


    def get_trained_model(self):
        print(f"blstm self.model_to_load:{self.model_to_load}")
        return self.model_to_load
    
    
    """
    Trains model based on training data
    """


    def train(self,args,starttime):
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
        # from History import LossHistory
        # history = LossHistory()

        # self.model.fit(self.X_train, self.y_train,validation_split=0.2,batch_size=self.batch_size,
        #                epochs=4,verbose=1,
        #                class_weight=self.class_weight)
        self.model.fit(self.X_train, self.y_train, validation_split=0.2,batch_size=self.batch_size, epochs=self.epochs,
                       class_weight=self.class_weight)
        predictions = (self.model.predict(self.X_test, batch_size=self.batch_size)).round()
        tn, fp, fn, tp = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(predictions, axis=1)).ravel()
        acc=(tp+tn)/(tp+fp+fn+tn)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = (2 * precision * recall) / (precision + recall)
        nni.report_final_result(f1)
        self.f1=f1
        endtime = datetime.datetime.now()
        self.alltime=(datetime.datetime.now()-starttime).seconds
        self.model_to_load=str(f1)+"_"+str(self.randomstate)+".h5"
        self.model.save_weights(self.model_to_load)
        print(f"saving the trained model in {self.model_to_load}")
        # self.model.save(self.name + "_whole_model.h5")

    """
    Tests accuracy of model based on test data
    Loads weights from file if no weights are attached to model object
    """
    def test(self):
        self.model.load_weights(self.model_to_load)
        predictions = (self.model.predict(self.X_test, batch_size=self.batch_size)).round()

        tn, fp, fn, tp = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(predictions, axis=1)).ravel()
        acc=(tp+tn)/(tp+tn+fp+fn)
        print("Accuracy is...", acc)
        recall = tp / (tp + fn)
        print('True recall is...', recall)
        precision = tp / (tp + fp)
        print('Precision is...', precision)
        f1 = (2 * precision * recall) / (precision + recall)
        print('F1 score is...',f1 )
        
        



        #写人excel
        # data = pd.DataFrame([[os.path.basename(self.name),self.randomstate,values[1],recall,precision,f1,tpr,fpr,tnr,fnr]])
        #
        # filename = str(self.name+"_result.xlsx")
        # writer = pd.ExcelWriter(filename)  # 写入Excel文件
        # data.to_excel(writer, index=False,header=["name","seed","acc","recall","precision","f1","tpr","fpr","tnr","fnr"],encoding="utf8")  # ‘page_1’是写入excel的sheet名
        # writer.save()
        # writer.close()
        
    """
    Tests accuracy of model based on all data (test data+ train data in this class)
    Loads weights from file if no weights are attached to model object
    """    
    def prediction(self):
        X=np.concatenate((self.X_test,self.X_train))
        y=np.concatenate((self.y_test,self.y_train))
        self.model.load_weights(self.model_to_load)
        predictions = (self.model.predict(X, batch_size=self.batch_size)).round()
        tn, fp, fn, tp = confusion_matrix(np.argmax(y, axis=1), np.argmax(predictions, axis=1)).ravel()
        acc=(tp+tn)/(tp+tn+fp+fn)
        print("Accuracy is...", acc)
        recall = tp / (tp + fn)
        print('True recall is...', recall)
        precision = tp / (tp + fp)
        print('Precision is...', precision)
        f1 = (2 * precision * recall) / (precision + recall)
        print('F1 score is...',f1 )
        

        def draw(acc,pre,recall,f1):
            # data = pd.read_excel(path)
            bar_width = 0.5
            # plt.bar(data.iloc[:, 0], data.iloc[:, 1],)
            # plt.show()
            x = np.array(["accuracy", "precision", "recall", "f1"])
            y = np.array([acc, pre, recall, f1])
            plt.bar(x,y,width=bar_width,color='#EEB0AF')
            plt.savefig('plot.png')  
            plt.show()
            
        draw(acc,precision,recall,f1)
        print("draw(acc,precision,recall,f1).....")
    
