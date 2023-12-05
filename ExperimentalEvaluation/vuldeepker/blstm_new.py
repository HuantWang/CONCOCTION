from __future__ import print_function

import warnings

from keras.utils import to_categorical
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight

warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, LeakyReLU
from keras.optimizers import Adamax

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
class BLSTM_NEW:
    # 精确率评价指标


    def __init__(self, X_train,X_test,y_train,y_test, name, batch_size=64):
        # vectors = np.stack(data.iloc[:, 0].values)
        # labels = data.iloc[:, 1].values
        # positive_idxs = np.where(labels == 1)[0]
        # negative_idxs = np.where(labels == 0)[0]
        # undersampled_negative_idxs = np.random.choice(negative_idxs, len(positive_idxs), replace=True)#从样本中随机选择size大小的元素
        # resampled_idxs = np.concatenate([positive_idxs, undersampled_negative_idxs])
        #
        # X_train, X_test, y_train, y_test = train_test_split(vectors[resampled_idxs, ], labels[resampled_idxs],
        #                                                     test_size=0.99, stratify=labels[resampled_idxs])
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)
        self.name = name
        self.batch_size = batch_size
        self.class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=y_train)
        model = Sequential()
        model.add(Bidirectional(LSTM(300), input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(300))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(Dense(300))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        # Lower learning rate to prevent divergence
        adamax = Adamax(lr=0.002)

        #########
        def pre(y_true, y_pred):
            TP = tf.reduce_sum(y_true * tf.round(y_pred))
            TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
            FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
            FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
            precision = TP / (TP + FP)
            return precision

        # 召回率评价指标
        def re(y_true, y_pred):
            TP = tf.reduce_sum(y_true * tf.round(y_pred))
            TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
            FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
            FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
            recall = TP / (TP + FN)
            return recall

        # F1-score评价指标
        def F1(y_true, y_pred):
            TP = tf.reduce_sum(y_true * tf.round(y_pred))
            TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
            FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
            FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            F1score = 2 * precision * recall / (precision + recall)
            return F1score

        model.compile(adamax, 'categorical_crossentropy', metrics=['accuracy'])

        self.model = model

    """
    Trains model based on training data
    """
    def train(self):
        #修改此处epoch
        self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, epochs=4, class_weight=self.class_weight)
        self.model.save_weights(self.name + "_model.h5")
        # self.model.save(self.name + "_whole_model.h5")


    """
    Tests accuracy of model based on test data
    Loads weights from file if no weights are attached to model object
    """
    def test(self):
        self.model.load_weights(self.name + "_model.h5")
        # self.model.load_weights("CWE-191_model.h5")
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
        print('F1 score is...', (2 * precision * recall) / (precision + recall))