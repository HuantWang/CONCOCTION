"""
python /home/ExperimentalEvaluation/vuldeepker/vuldeepecker.py --data_path /home/ExperimentalEvaluation/vuldeepker/datas --mode train
python /home/ExperimentalEvaluation/vuldeepker/vuldeepecker.py --data_path /home/ExperimentalEvaluation/vuldeepker/datas --mode pre --model_to_load /home/ExperimentalEvaluation/vuldeepker/0.9691056910569105_85158361.h5
"""
import os
import pandas
from clean_gadget import clean_gadget
from vectorize_gadget import GadgetVectorizer
from blstm import BLSTM
from blstm_new import BLSTM_NEW
from keras.models import load_model
import numpy as np
import time
import nni

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
import torch
from transformers import AutoTokenizer, AutoModel
import datetime

def get_vectors_df_new(fpath):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    def findAllFile(dir):
        for root, ds, fs in os.walk(dir):
            for f in fs:
                yield root, f

    data_path = fpath
    y = []
    gap = " "
    Graph_length = 50
    X_feature = {
        "name": {},
        "X_Code": {},
        "X_trace": {},
        "X_testcase": {},
        "X_Graph": {},
        "X_Node": {},
        "X_dynamic": {},
        "label": {},
    }
    X_feature["name"] = []
    X_feature["X_Code"] = []
    X_feature["X_trace"] = []
    X_feature["X_testcase"] = []
    X_feature["X_Graph"] = []
    X_feature["X_Node"] = []
    X_feature["X_dynamic"] = []
    X_feature["label"] = []
    vectors = []
    # for root, file in tqdm(findAllFile(data_path), desc='dirs'):
    for root, file in findAllFile(data_path):
        if file.endswith(".txt"):
            flag = "none"
            file_path = os.path.join(root, file)

            X_Code_Single = []
            X_Graph_Single = np.zeros([Graph_length, Graph_length])
            X_trace_Single = []
            X_testcase_single = []
            X_Node_Singe = []
            X_dynamic_single = []
            f = open(file_path)
            try:
                for line in f:
                    if line == "-----label-----\n":
                        flag = "label"
                        continue
                    if line == "-----code-----\n":
                        flag = "code"
                        continue
                    if line == "-----children-----\n":
                        flag = "children"
                        continue
                    if line == "-----nextToken-----\n":
                        flag = "nextToken"
                        continue
                    if line == "-----computeFrom-----\n":
                        flag = "computeFrom"
                        continue
                    if line == "-----guardedBy-----\n":
                        flag = "guardedBy"
                        continue
                    if line == "-----guardedByNegation-----\n":
                        flag = "guardedByNegation"
                        continue
                    if line == "-----lastLexicalUse-----\n":
                        flag = "lastLexicalUse"
                        continue
                    if line == "-----jump-----\n":
                        flag = "jump"
                        continue
                    if line == "=======testcase========\n":
                        flag = "testcase"
                        continue
                    if line == "=========trace=========\n":
                        flag = "trace"
                        continue
                    if (
                            line == "-----attribute-----\n"
                            or line == "----------------dynamic----------------\n"
                    ):
                        flag = "next"
                        continue
                    if line == "-----ast_node-----\n":
                        flag = "ast_node"
                        continue
                    if line == "=======================\n":
                        break
                    if flag == "next":
                        continue
                    if flag == "label":
                        y = line.split()
                        continue
                    if flag == "code":
                        X_Code_line = line.split("\n")[0]
                        X_Code_Single = X_Code_Single + [X_Code_line]
                        continue
                    if flag == "children":
                        num_1 = int(line.split()[0].split(",")[0])
                        num_2 = int(line.split()[0].split(",")[1])
                        if num_2 < Graph_length and num_1 < Graph_length:
                            X_Graph_Single[num_1 - 1, num_2 - 1] = 1
                        else:
                            continue
                        continue
                    if flag == "nextToken":
                        num_1 = int(line.split()[0].split(",")[0])
                        num_2 = int(line.split()[0].split(",")[1])
                        X_Graph_Single[num_1 - 1, num_2 - 1] = 1
                        continue
                    if flag == "computeFrom":
                        num_1 = int(line.split()[0].split(",")[0])
                        num_2 = int(line.split()[0].split(",")[1])
                        if num_2 < Graph_length and num_1 < Graph_length:
                            X_Graph_Single[num_1 - 1, num_2 - 1] = 1
                        else:
                            continue
                        continue
                    if flag == "guardedBy":
                        num_1 = int(line.split()[0].split(",")[0])
                        num_2 = int(line.split()[0].split(",")[1])
                        if num_2 < Graph_length and num_1 < Graph_length:
                            X_Graph_Single[num_1 - 1, num_2 - 1] = 1
                        else:
                            continue
                        continue
                    if flag == "guardedByNegation":
                        num_1 = int(line.split()[0].split(",")[0])
                        num_2 = int(line.split()[0].split(",")[1])
                        if num_2 < Graph_length and num_1 < Graph_length:
                            X_Graph_Single[num_1 - 1, num_2 - 1] = 1
                        else:
                            continue
                        continue
                    if flag == "lastLexicalUse":
                        num_1 = int(line.split()[0].split(",")[0])
                        num_2 = int(line.split()[0].split(",")[1])
                        if num_2 < Graph_length and num_1 < Graph_length:
                            X_Graph_Single[num_1 - 1, num_2 - 1] = 1
                        else:
                            continue
                        continue
                    if flag == "jump":
                        num_1 = int(line.split()[0].split(",")[0])
                        num_2 = int(line.split()[0].split(",")[1])
                        if num_2 < Graph_length and num_1 < Graph_length:
                            X_Graph_Single[num_1 - 1, num_2 - 1] = 1
                        else:
                            continue
                        continue
                    if flag == "ast_node":
                        X_Code_line = line.split("\n")[0]
                        X_Node_Singe = X_Node_Singe + [X_Code_line]
                        continue
                    if flag == "testcase":
                        X_Code_line = line.split("\n")[0]
                        X_testcase_single = X_testcase_single + [X_Code_line]
                        X_dynamic_single = X_dynamic_single + [X_Code_line]
                    if flag == "trace":
                        X_Code_line = line.split("\n")[0]
                        X_trace_Single = X_trace_Single + [X_Code_line]
                        X_dynamic_single = X_dynamic_single + [X_Code_line]
                f.close()
            except:
                print("please delete the file " + file)

            # X_feature["name"].append(file_path)
            X_feature["X_Code"].append(gap.join(X_Code_Single))
            # X_feature["X_trace"].append(gap.join(X_trace_Single).split())
            # X_feature["X_testcase"].append(gap.join(X_testcase_single).split())
            # X_feature["X_Graph"].append(X_Graph_Single)
            # X_feature["X_Node"].append(gap.join(X_Node_Singe))
            # X_feature["X_dynamic"].append(gap.join(X_dynamic_single).split())
            X_feature["label"].append(int(y[0]))

            nl_tokens = tokenizer.tokenize(gap.join(X_Code_Single)+"safe word " * 64)
            tokens_ids=tokenizer.convert_tokens_to_ids(nl_tokens)[:50]
            context_embeddings = model(torch.tensor(tokens_ids)[None, :])[0]

            context_embeddings=torch.squeeze(context_embeddings).detach().numpy()[:,:50]
            row = {"gadget": context_embeddings, "val": int(y[0])}
            vectors.append(row)
    # vectorizer.train_model()
    # for gadget,label in zip(X_feature["X_Code"],X_feature["label"]):
    #     vector = vectorizer.vectorize(gadget)
    #     row = {"gadget": vector, "val": label}
    #     vectors.append(row)
    df = pandas.DataFrame(vectors)
    return df

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


def  get_default_parameters():
    params = {
        'batch_size': 64,
        'epochs': 5,
        'lr': 0.02,
        'dropout': 0.5,
        'active': 'sigmoid'
    }
    return params
"""
Gets filename, either loads vector DataFrame if exists or creates it if not
Instantiate neural network, pass data to it, train, test, print accuracy
"""
def main1():

    # 修改数据路径

    filedir = './datas'
    # test
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--model_to_load", type=str,help="model_to_load", required=False)
    PARSER.add_argument("--data_path", type=str,  help="data path", required=True)
    PARSER.add_argument("--mode", type=str,  help="train or test", required=False)
    PARSER.add_argument("--batch_size", type=int, default=64, help="batch size", required=False)
    PARSER.add_argument("--epochs", type=int, default=2, help="Train epochs", required=False)
    PARSER.add_argument("--lr", type=int, default=0.001, help="Number of train samples to be used, maximum 60000",
                        required=False)
    PARSER.add_argument("--dropout", type=int, default=0.5,
                        required=False)
    PARSER.add_argument("--active", type=str, default='softmax',
                        required=False)
    ARGS, UNKNOWN = PARSER.parse_known_args()

    try:
        # get parameters from tuner
        try:
            RECEIVED_PARAMS = nni.get_next_parameter()
            args = PARSER.parse_args()
            args = vars(args)
            args.update(RECEIVED_PARAMS)
            args = argparse.Namespace(**args)
        except Exception as e:
            args = PARSER.parse_args()
            RECEIVED_PARAMS = nni.get_next_parameter()
        filedir = args.data_path
        if args.mode=='pre':
            df = get_vectors_df_new(filedir)
            blstm = BLSTM(df, args)
            blstm.prediction()
            return
            
        df = get_vectors_df_new(filedir)
        blstm = BLSTM(df, args)
        starttime=datetime.datetime.now()
        blstm.train(args,starttime)
        endtime=datetime.datetime.now()
        blstm.test()

    except Exception as e:
        raise
    # name = os.path.basename(filedir) + "-" + str(round(time.time()))
    # base = os.path.join("result",name)
    # vector_filename = base + "_gadget_vectors.pkl"

    # df.to_pickle(vector_filename)


    # os.remove(vector_filename)
    # os.remove(base + "_model.h5")

def main2():
    """测试机和训练集分开"""
    # base = os.path.splitext(os.path.basename(filename))[0]

    #=============train dataset=============================
    #实验四：增加数据

    filedir_1 = r"./Data/CWE-200"

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

    base_1 = "CWE-200_train"
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
    blstm.test()

    os.remove(vector_filename)
    os.remove(base_1 + "_model.h5")


if __name__ == "__main__":

    #cmd
    #python vuldeepecker.py -f "CWE-200"


    main1()   #main1为原来的方法，用于自测
    # main2() #训练集和测试集分开开开开开
