"""
Interface to VulDeePecker project
"""
import argparse
import random
import sys
import os

import joblib
import pandas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report

from clean_gadget import clean_gadget
from vectorize_gadget import GadgetVectorizer
import re_bilstm
from keras.models import load_model
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.metrics import *


"""
此代码用于TDSC（Lin）的自测的实验
当进行使用时，修改数据路径filedir_1
"""

"""
Parses gadget file to find individual gadgets
Yields each gadget as list of strings, where each element is code line
Has to ignore first line of each gadget, which starts as integer+space
At the end of each code gadget is binary value
    This indicates whether or not there is vulnerability in that gadget
"""


def parse_file(filename):
    for file_name in os.listdir(filename):
        with open(os.path.join(filename, file_name), "r", encoding="utf8") as file:
            # with open(filename, "r", encoding="utf8") as file:
            gadget = []
            gadget_val = 0
            for line in file:
                stripped = line.strip()
                if not stripped:
                    continue
                if "^" * 15 in line and gadget:
                    yield clean_gadget(gadget), gadget_val
                    gadget = []
                elif stripped.split()[0].isdigit():
                    if gadget:
                        # Code line could start with number (somehow)
                        if stripped.isdigit():
                            gadget_val = int(stripped)
                        else:
                            gadget.append(stripped)
                else:
                    gadget.append(stripped)


"""
Uses gadget file parser to get gadgets and vulnerability indicators
Assuming all gadgets can fit in memory, build list of gadget dictionaries
    Dictionary contains gadgets and vulnerability indicator
    Add each gadget to GadgetVectorizer
Train GadgetVectorizer model, prepare for vectorization
Loop again through list of gadgets
    Vectorize each gadget and put vector into new list
Convert list of dictionaries to dataframe when all gadgets are processed
"""


def get_vectors_df(filename, vector_length=100):
    gadgets = []
    count = 0
    vectorizer = GadgetVectorizer(vector_length)
    for gadget, val in parse_file(filename):
        count += 1
        print("Collecting gadgets...", count, end="\r")
        vectorizer.add_gadget(gadget)
        row = {"gadget": gadget, "val": val}
        gadgets.append(row)
    print('Found {} forward slices and {} backward slices'
          .format(vectorizer.forward_slices, vectorizer.backward_slices))
    print()
    print("Training model...", end="\r")
    vectorizer.train_model()
    print()
    vectors = []
    count = 0
    for gadget in gadgets:
        count += 1
        print("Processing gadgets...", count, end="\r")
        vector = vectorizer.vectorize(gadget["gadget"])
        row = {"vector": vector, "val": gadget["val"]}
        vectors.append(row)
    print()
    df = pandas.DataFrame(vectors)
    return df


"""
Gets filename, either loads vector DataFrame if exists or creates it if not
Instantiate neural network, pass data to it, train, test, print accuracy
"""

#
import logging
import time
import os

def main2(f,r):

    filedir_1 = f
    # test

    name = os.path.basename(filedir_1) + "-" + str(round(time.time()))
    base_mudi = os.path.join(r"F:\xrz\科研\3静态漏洞检测\contrastExperiment\2model\TDSC\re_result", name)
    base_mudi = os.path.join(r"C:\Users\Administrator\Desktop\3Experiment\3Experiment\TDSC\Data\re_result", name)

    # vector_filename = base + "_gadget_vectors.pkl"
    vector_filename = r + "_gadget_vectors.pkl"
    vector_length = 100
    if os.path.exists(vector_filename):
        df = pandas.read_pickle(vector_filename)
    else:
        df = get_vectors_df(filedir_1, vector_length)
        df.to_pickle(vector_filename)

    # 获取seed
    data_frame = pd.read_excel(r + "_result.xlsx", sheet_name="Sheet1")
    seed = data_frame['seed'][0]

    # vectors = np.stack(df.iloc[:, 1].values)
    # labels = df.iloc[:, 0].values

    blstm = re_bilstm.BLSTM(df,seed, name=base_mudi)
    # blstm = BLSTM(df,name=base)
    data, label = blstm.train_self(r)
    # print("a")

    #seed
    train_X, test_X, train_y, test_y = train_test_split(data,
                                                        label,
                                                        test_size=0.4,  # 1
                                                        random_state=seed)

    # clf = RandomForestClassifier(bootstrap=True, class_weight='balanced',  # class_weight={0:1, 1:4},
    #                              criterion='entropy', max_depth=40, max_features='auto',
    #                              max_leaf_nodes=None, min_impurity_decrease=0.0,
    #                              min_impurity_split=None, min_samples_leaf=3,
    #                              min_samples_split=4, min_weight_fraction_leaf=0.0,
    #                              n_estimators=8000, oob_score=False, random_state=None,
    #                              verbose=1, warm_start=False, n_jobs=-1)


    from sklearn.model_selection import validation_curve

    # train_sizes, train_scores, valid_scores, fit_times, score_times = learning_curve(
    #     clf, train_X, train_y, scoring="accuracy", return_times=True)

    # train_sizes, train_scores, valid_scores, fit_times, score_times = learning_curve(
    #     clf, train_X, train_y, scoring="accuracy", return_times=True, n_jobs=2,
    #     train_sizes=np.linspace(0.1, 1.0, 150))

    # train_scores = np.array(train_scores)
    # train_sizes = np.array(train_sizes)
    # valid_scores = np.array(valid_scores)
    # fit_times = np.array(fit_times)
    # score_times = np.array(score_times)

    # train_scores_mean = np.mean(train_scores, axis=1)  # 将训练得分集合按行的到平均值
    # train_scores_std = np.std(train_scores, axis=1)  # 计算训练矩阵的标准方差
    # valid_scores_mean = np.mean(valid_scores, axis=1)
    # valid_scores_std = np.std(valid_scores, axis=1)
    #
    # train_sizes = train_sizes.reshape(train_sizes.size, 1)
    # train_scores = train_scores[:, 0].reshape(train_scores[:, 0].size, 1)
    # valid_scores = valid_scores[:, 0].reshape(valid_scores[:, 0].size, 1)
    # fit_times = fit_times[:, 0].reshape(fit_times[:, 0].size, 1)
    # score_times = score_times[:, 0].reshape(score_times[:, 0].size, 1)

    # train_scores_mean = train_scores_mean.reshape(train_scores_mean.size, 1)
    # train_scores_std = train_scores_std.reshape(train_scores_std.size, 1)
    # valid_scores_mean = valid_scores_mean.reshape(valid_scores_mean.size, 1)
    # valid_scores_std = valid_scores_std.reshape(valid_scores_std.size, 1)
    #
    # data = np.concatenate((train_sizes, train_scores, valid_scores, fit_times, score_times, train_scores_mean,
    #                        train_scores_std, valid_scores_mean, valid_scores_std), axis=1)
    # 写入excel
    # import pandas as pd
    # data = pd.DataFrame(data)
    # filename = "Excel//" + str(base_mudi + "-" + str(time.time()) + ".xlsx")
    # writer = pd.ExcelWriter(filename)  # 写入Excel文件
    # data.to_excel(writer, index=False,
    #               header=["train_sizes", "train_scores", "valid_scores", "fit_times", "score_times",
    #                       "train_scores_mean", "train_scores_std", "valid_scores_mean",
    #                       "valid_scores_std"])  # ‘page_1’是写入excel的sheet名
    # writer.save()
    # writer.close()

    # import matplotlib.pyplot as plt
    # plt.xlabel("Training examples")  # 两个标题
    # plt.ylabel("Score")
    #
    # plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
    #                  color='r')
    # # plt.fill_between()函数会把模型准确性的平均值的上下方差的空间里用颜色填充。
    # plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1,
    #                  color='g')
    # plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    # # 然后用plt.plot()函数画出模型准确性的平均值
    # plt.plot(train_sizes, valid_scores_mean, 'o-', color='g', label='Cross_validation score')
    # plt.legend(loc='best')  # 显示
    # plt.show()

    # 训练
    # clf.fit(train_X, train_y)

    #读取模型
    clf = joblib.load(r + "_model.pkl")

    y_pred = clf.predict(test_X)

    print(confusion_matrix(test_y, y_pred))
    print(classification_report(test_y.ravel(), y_pred))
    # print(CWE)
    # os.remove(vector_filename)

    pre = precision_score(test_y.ravel(), y_pred, average="micro")
    acc = accuracy_score(test_y.ravel(), y_pred)
    f1 = f1_score(test_y.ravel(), y_pred, average="micro")
    recall = recall_score(test_y.ravel(), y_pred, average="micro")
    tnr = (acc*(pre + recall - pre*recall)-pre*recall)/(recall - 2*pre*recall + acc*pre)
    fpr = 1-tnr
    tpr = recall
    fnr = 1-tpr
    #写入excel
    # 写人excel
    data = pd.DataFrame(
        [[os.path.basename(name),os.path.basename(r), seed, acc, recall, pre, f1, tpr, fpr, tnr, fnr]])

    filename = str(base_mudi + "_result.xlsx")
    writer = pd.ExcelWriter(filename)  # 写入Excel文件
    data.to_excel(writer, index=False,
                  header=["name", "rename","seed", "acc", "recall", "precision", "f1", "tpr", "fpr", "tnr", "fnr"],
                  encoding="utf8")  # ‘page_1’是写入excel的sheet名
    writer.save()
    writer.close()


if __name__ == "__main__":
    # 复现实验的代码
    # cmd
    # python re_TDSC_self.py -f "Data/CWE-200" -r "F:\xrz\科研\3静态漏洞检测\contrastExperiment\2model\TDSC\result\CWE-200-1649658234"
    parser = argparse.ArgumentParser()
    parser.description = 'please enter data path:'
    parser.add_argument("-f", "--fp", help="data file path", dest="f", type=str)
    parser.add_argument("-r", "--r", help="recurrent experiment path", dest="r", type=str)
    args = parser.parse_args()

    main2(args.f, args.r)  # main1为原来的方法，用于自测
