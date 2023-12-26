from __future__ import print_function

import warnings

import joblib
import keras
from keras import optimizers, Model
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
# from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import compute_class_weight

warnings.filterwarnings("ignore")

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, LeakyReLU, Embedding, GlobalMaxPooling1D
import random
import nni
import datetime

from sklearn.model_selection import train_test_split

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
import keras_metrics
class BLSTM:
    def __init__(self, data,args):
        self.randomstate = random.randint(1, pow(2, 32))
        vectors = np.stack(data.iloc[:, 0].values)
        labels = data.iloc[:, 1].values
        positive_idxs = np.where(labels == 1)[0]
        negative_idxs = np.where(labels == 0)[0]
        # undersampled_negative_idxs = np.random.choice(negative_idxs, len(positive_idxs),
        #                                               replace=True)  # 从样本中随机选择size大小的元素
        undersampled_negative_idxs = negative_idxs[0:len(positive_idxs)]
        resampled_idxs = np.concatenate([positive_idxs, undersampled_negative_idxs])

        X_train, X_test, y_train, y_test = train_test_split(vectors[resampled_idxs,], labels[resampled_idxs],
                                                            test_size=0.2, stratify=labels[resampled_idxs],random_state=self.randomstate)
        self.vector = vectors
        self.label = labels
        self.X_train = X_train
        self.X_test = X_test
        self.f1=0
        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=labels)
        # self.class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)
        # a=X_train.shape[1]
        model = Sequential()
        model.add(Bidirectional(LSTM(64), input_shape=(X_train.shape[1], X_train.shape[2])))
        # model.add(Bidirectional(LSTM(64), input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(64, activation='relu'))
        # aintermediate_layer_model = Model(inputs=model.input,
        #                                  outputs=model.get_layer(index=1).output)
        # aintermediate_output = aintermediate_layer_model.predict(vectors)  # a=model.layers[0].output
        model.add(Dense(32))
        model.add(Dropout(args.dropout))
        model.add(Dense(2, activation=args.active))
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizers.SGD(lr=args.lr, decay=0.01 / 50, nesterov=True),
                      metrics=[keras_metrics.precision(),keras_metrics.recall(),keras_metrics.f1_score(),'accuracy'])

        self.model = model
        if args.mode=='pre':
            self.model_to_load=args.model_to_load

    def get_trained_model(self):
        print(f"blstm self.model_to_load:{self.model_to_load}")
        return self.model_to_load


    """
    Trains model based on training data
    """
    def train(self,starttime):
        self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs, class_weight=self.class_weight)
        predictions = (self.model.predict(self.X_test, batch_size=self.batch_size)).round()
        tn, fp, fn, tp = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(predictions, axis=1)).ravel()
        acc = (tp + tn) / (tp + fp + fn + tn)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = (2 * precision * recall) / (precision + recall)
        nni.report_final_result(f1)
        self.f1 = f1
        endtime = datetime.datetime.now()
        alltime= (endtime - starttime).seconds
        self.alltime=alltime
        self.model_to_load=str(alltime)+"_" +str(f1) + "_" + str(self.randomstate) + ".h5"
        try:
            self.model.save(self.model_to_load)
        except:
            pass
        # self.model.save("model.h5")

        # self.model.save(self.name + "_whole_model.h5")

    def train_self(self):
        # self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, epochs=50, class_weight=self.class_weight)

        aintermediate_layer_model = Model(inputs=self.model.input,
                                          outputs=self.model.get_layer(index=1).output)

        joblib.dump(aintermediate_layer_model, self.name + "_aintermediate_layer_model.pkl")

        aintermediate_output = aintermediate_layer_model.predict(self.vector,
                                                                 batch_size=self.batch_size)  # a=model.layers[0].output


        # inter_model = joblib.load(r"F:\xrz\科研\3静态漏洞检测\contrastExperiment\2model\TDSC\re_result\CWE-200-1649681804_aintermediate_layer_model.pkl")
        # aintermediate_output = inter_model.predict(self.vector,
        #                                                          batch_size=self.batch_size)  # a=model.layers[0].output
        return aintermediate_output,self.label
    """
    Tests accuracy of model based on test data
    Loads weights from file if no weights are attached to model object
    """
    def test(self):

        # self.model.load_weights(self.name + "_model.h5")
        # self.model.compile(loss='binary_crossentropy',
        #               optimizer=optimizers.SGD(lr=0.01, decay=0.01 / 50, nesterov=True),
        #               metrics=[keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score(), 'accuracy'])
        #
        # return predictions
        #self.model=keras.models.load_model("model.h5")
        self.model.load_weights(self.model_to_load)




        # self.model.load_weights(str(self.alltime)+"_" +str(self.f1)+"_"+str(self.randomstate)+".h5")
        aintermediate_layer_model = Model(inputs=self.model.input,
                                          outputs=self.model.get_layer(index=1).output)
        aintermediate_output = aintermediate_layer_model.predict(self.vector, batch_size=self.batch_size)  # a=model.layers[0].output
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
        print('F1 score is...', f1)


        return aintermediate_output,self.label
    
    def prediction(self):
        print("prediction....")
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
        
        def result_log(acc,precision,recall,f1):
            file_path="./result.log"
            with open(file_path, 'w') as f:
                f.writelines(f"acc,{acc}\n")
                f.writelines(f"pre,{precision}\n")
                f.writelines(f"recall,{recall}\n")
                f.writelines(f"f1,{f1}")
                
        result_log(acc,precision,recall,f1)
    