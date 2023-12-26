"""
Interface to VulDeePecker project
"""
import os
import pandas
from clean_gadget import clean_gadget
from vectorize_gadget import GadgetVectorizer
from re_blstm import BLSTM
from blstm_new import BLSTM_NEW
from keras.models import load_model
import numpy as np
import time
import pandas as pd
import argparse
"""
此代码用于vuldeepecker和uvuldeepecker的对比实验
main1是自测，其中blatm.py中train函数的epoch=4是vuldeepecker，epoch=10是uvuldeepecker
main2是训练集测试集分开，其中blatm_new.py中train函数的epoch=4是vuldeepecker，epoch=10是uvuldeepecker

当使用main1进行自测时，修改数据路径filedir
当使用main2进行训练集和测试集分开时，修改训练集路径filedir_1，修改测试集路径filedir_2

注：可以在vectorize_gadget.py的train_model中指定字典，也可以不指定
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
            next(file)
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
        row = {"gadget" : gadget, "val" : val}
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
        row = {"vector" : vector, "val" : gadget["val"]}
        vectors.append(row)
    print()
    df = pandas.DataFrame(vectors)
    return df
            
"""
Gets filename, either loads vector DataFrame if exists or creates it if not
Instantiate neural network, pass data to it, train, test, print accuracy
"""
def main1(f,r):

    # base = os.path.splitext(os.path.basename(filename))[0]
    #训练
    # CWE = "CWE-787"
    # filedir = r"D:\XRZ\数据\sard\C\切片"+"\\"+CWE+"\\"+CWE

    # #修改数据路径
    # dirpath = r"Data"
    # # dirpath = r"H:\XRZ\漏洞检测\Data\sard\C\切片\CWE-119\merge"
    # cwes = os.listdir(dirpath)
    # for cwe in cwes:
    #     for i in range(1,2):
    #         filedir = os.path.join(dirpath, cwe)
    #
    #         # filedir = r"D:\XRZ\数据\sard\C\切片\CWE-668\5\train" #c
    #         # filedir = r"D:\XRZ\数据\sard\JAVA\切片\CWE-668\5\train" #java
    #         # filedir = r"D:\XRZ\数据\sard\C\切片\CWE-668\5\test" #c
    #         # filedir = r"D:\XRZ\数据\sard\JAVA\切片\CWE-668\5\test" #java
    #         # test
    #
    #         base = cwe
    #         vector_filename = base + "_gadget_vectors.pkl"
    #         vector_length = 100
    #         if os.path.exists(vector_filename):
    #             df = pandas.read_pickle(vector_filename)
    #         else:
    #             df = get_vectors_df(filedir, vector_length)
    #             df.to_pickle(vector_filename)
    #         blstm = BLSTM(df, name=base)
    #         blstm.train()
    #         blstm.test()
    #
    #         os.remove(vector_filename)
    #         os.remove(base + "_model.h5")


    # 修改数据路径

    # filedir = r"F:\xrz\科研\3静态漏洞检测\contrastExperiment\1dataset\devign\1slice\FFmpeg\test"
    filedir = f
    # test

    name = os.path.basename(filedir) + "-" + str(round(time.time()))
    base = os.path.join(r"F:\xrz\科研\3静态漏洞检测\contrastExperiment\2model\vuldeepecker\re_result",name)
    # vector_filename = base + "_gadget_vectors.pkl"
    vector_filename = r + "_gadget_vectors.pkl"
    vector_length = 100
    if os.path.exists(vector_filename):
        df = pandas.read_pickle(vector_filename)
    else:
        df = get_vectors_df(filedir, vector_length)
        df.to_pickle(vector_filename)

    #获取seed
    data_frame = pd.read_excel(r+"_result.xlsx", sheet_name="Sheet1")
    seed = data_frame['seed'][0]
    blstm = BLSTM(df,seed, name=base )
    # blstm.train()
    modelname = r + "_model.h5"
    blstm.test(modelname)

    # os.remove(vector_filename)
    # os.remove(base + "_model.h5")

def main2():
    """测试机和训练集分开"""
    # base = os.path.splitext(os.path.basename(filename))[0]

    #=============train dataset=============================
    #实验四：增加数据

    filedir_1 = r"Data/CWE-670"

    # filedir_1 = r"D:\XRZ\补充数据实验\data\CWE-670\sard"
    # filedir_1 = r"D:\XRZ\补充数据实验\data\CWE-670\sard_and_github1"
    # 实验五：迁移
    # filedir_1 = r"D:\XRZ\数据\sard\C\切片\CWE-077\5\train" #c

    # filedir_1 = r"D:\XRZ\数据\sard\JAVA\切片\CWE-077\5\train" #java
    # filedir_1 = r"D:\XRZ\数据\sard\JAVA\切片\CWE-074\5\train" #java

    #添加测试集语言
    # filedir_1 = r"D:\XRZ\数据\sard\mix\CWE-191\cjava"
    # filedir_1 = r"D:\XRZ\数据\sard\mix\CWE-191\javac"
    # filedir_1 = r"D:\XRZ\数据\sard\mix\CWE-191\11"

    #实验六：测试集比例
    # filedir_1 = r"D:\XRZ\补充数据实验\data\CWE-704\sard"

    # filedir_1 = r"D:\XRZ\补充数据实验\data\CWE-404\sard\sard"

    base_1 = "CWE-191_train"
    vector_filename = base_1 + "_gadget_vectors.pkl"
    vector_length = 100
    if os.path.exists(vector_filename):
        df = pandas.read_pickle(vector_filename)
    else:
        df = get_vectors_df(filedir_1, vector_length)
        df.to_pickle(vector_filename)
    vectors = np.stack(df.iloc[:, 0].values)
    labels = df.iloc[:, 1].values
    positive_idxs = np.where(labels == 1)[0]
    negative_idxs = np.where(labels == 0)[0]
    undersampled_negative_idxs = np.random.choice(negative_idxs, len(positive_idxs), replace=True)  # 从样本中随机选择size大小的元素
    resampled_idxs = np.concatenate([positive_idxs, undersampled_negative_idxs])
    X_train=vectors[resampled_idxs,]
    y_train=labels[resampled_idxs]

    os.remove(vector_filename)

    #=======================test dataset==========================

    # filedir_2 = r"D:\XRZ\补充数据实验\data\CWE-670\github\str分类\github2"

    # filedir_2 = r"D:\XRZ\数据\sard\C\切片\CWE-077\5\test"  # c
    # filedir_2 = r"D:\XRZ\数据\sard\JAVA\切片\CWE-077\5\java"  # java

    # filedir_2 = r"D:\XRZ\数据\sard\JAVA\切片\CWE-074\5\test"  # java
    # filedir_2 = r"D:\XRZ\数据\sard\jpmix\CWE-074\第一次\php"  # php

    # filedir_2 = r"D:\XRZ\补充数据实验\data\CWE-665\github\分类\6\1_4"
    # filedir_2 = r"D:\XRZ\补充数据实验\data\CWE-074\github\分类\6\1_9"

    # filedir_2 = r"D:\XRZ\补充数据实验\data\CWE-670\github\gra分类\第一次\merge"

    filedir_2 = r"Data/CWE-200"

    base_2 = "CWE-191_test"
    vector_filename = base_2 + "_gadget_vectors.pkl"
    vector_length = 100
    if os.path.exists(vector_filename):
        df = pandas.read_pickle(vector_filename)
    else:
        df = get_vectors_df(filedir_2, vector_length)
        df.to_pickle(vector_filename)
    vectors = np.stack(df.iloc[:, 0].values)
    labels = df.iloc[:, 1].values
    # x = set(labels)
    positive_idxs = np.where(labels == 1)[0]
    negative_idxs = np.where(labels == 0)[0]
    undersampled_negative_idxs = np.random.choice(negative_idxs, len(positive_idxs), replace=True)  # 从样本中随机选择size大小的元素
    resampled_idxs = np.concatenate([positive_idxs, undersampled_negative_idxs])
    X_test = vectors[resampled_idxs,]
    y_test = labels[resampled_idxs]
    blstm = BLSTM_NEW(X_train,X_test,y_train,y_test, name=base_1)
    # blstm = BLSTM(df,name=base)
    blstm.train()
    # blstm.test()

    os.remove(vector_filename)
    os.remove(base_1 + "_model.h5")


if __name__ == "__main__":

    #复现实验的代码
    # cmd
    # python re_vuldeepecker.py -f "CWE-200" -r "CWE-200-1649654717"
    parser = argparse.ArgumentParser()
    parser.description = 'please enter data path:'
    parser.add_argument("-f", "--fp", help="data file path", dest="f", type=str)
    parser.add_argument("-r", "--r", help="recurrent experiment path", dest="r", type=str)
    args = parser.parse_args()

    main1(args.f,args.r)   #main1为原来的方法，用于自测
    # main2() #训练集和测试集分开开开开开
