"""
/root/anaconda3/envs/vuldeepecker1/bin/python TDSC.py --data_path /home/ExperimentalEvaluation/data/github_0.6_new/test --model_to_load /home/ExperimentalEvaluation/TDSC/7_0.6796116504854369_759219936.h5 --mode pre
/root/anaconda3/envs/vuldeepecker1/bin/python TDSC.py --data_path /home/ExperimentalEvaluation/data/github_0.6_new/train  --mode train
"""
import sys
import os
import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import confusion_matrix, classification_report

from clean_gadget import clean_gadget
from vectorize_gadget import GadgetVectorizer
import bilstm
from keras.models import load_model
import numpy as np
import nni
import datetime
import argparse

"""
此代码用于TDSC（Lin）的训练集和测试集分开的实验
当进行使用时，修改训练集路径filedir_1，修改测试集路径filedir_2
"""

"""
Parses gadget file to find individual gadgets
Yields each gadget as list of strings, where each element is code line
Has to ignore first line of each gadget, which starts as integer+space
At the end of each code gadget is binary value
    This indicates whether or not there is vulnerability in that gadget
    
解析小工具文件以查找单个小工具
将每个小工具生成为字符串列表，其中每个元素都是代码行
必须忽略每个小工具的第一行，以整数+空格开头
每个代码小工具的末尾都是二进制值
    这表明该小工具中是否存在漏洞
"""

seed=123456

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

使用小工具文件解析器获取小工具和漏洞指标
假设所有小工具都可以放入内存，构建小工具字典列表
    字典包含小工具和漏洞指示器
    将每个小工具添加到 GadgetVectorizer 训练 GadgetVectorizer 模型，准备向量化 再次循环遍历小工具列表
    向量化每个小工具并将向量放入新列表 处理所有小工具时将字典列表转换为数据框
"""
import torch
from transformers import AutoTokenizer, AutoModel
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base").to(device)

def get_vectors_df_new(fpath):
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
            context_embeddings = model(torch.tensor(tokens_ids)[None, :].to(device))[0]

            context_embeddings=torch.squeeze(context_embeddings).detach().cpu().numpy()[:,:50]
            row = {"gadget": context_embeddings, "val": int(y[0])}
            vectors.append(row)
    # vectorizer.train_model()
    # for gadget,label in zip(X_feature["X_Code"],X_feature["label"]):
    #     vector = vectorizer.vectorize(gadget)
    #     row = {"gadget": vector, "val": label}
    #     vectors.append(row)
    df = pandas.DataFrame(vectors)
    print("a")
    return df


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

def  get_default_parameters():
    params = {
        'batch_size': 64,
        'epochs': 2,
        'lr': 0.02,
        'dropout': 0.5,
        'active': 'sigmoid'
    }
    return params


def main():
    
    base = r"C"
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--model_to_load", type=str,help="model_to_load", required=False)
    PARSER.add_argument("--data_path", type=str,  help="data path", required=True)
    PARSER.add_argument("--mode", type=str,  help="train or test", required=False)
    PARSER.add_argument("--batch_size", type=int, default=64, help="batch size", required=False)
    PARSER.add_argument("--epochs", type=int, default=5, help="Train epochs", required=False)
    PARSER.add_argument("--lr", type=int, default=0.02, help="Number of train samples to be used, maximum 60000",
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
            blstm = bilstm.BLSTM(df,args)
            blstm.prediction()
            return
            
        df = get_vectors_df_new(filedir)
        blstm = bilstm.BLSTM(df,args)
        starttime = datetime.datetime.now()
        blstm.train(starttime)
        data, label = blstm.test()

        # pre
        args.mode='pre'
        args.model_to_load = str(blstm.get_trained_model())
        df = get_vectors_df_new('/home/ExperimentalEvaluation/data/github_0.6_new/test')
        blstm2 = bilstm.BLSTM(df, args)
        blstm2.prediction()
    except Exception as e:
        raise
    # 源语言/Sard的目录
    # filedir_1 = r"D:\XRZ\补充数据实验\data\CWE-670\sard"
    # filedir_1 = r"F:\xrz\科研\3静态漏洞检测\contrastExperiment\2model\TDSC\Data\CWE-200"
    filedir = r"./datas"
    base = r"C"
    # base_source = os.path.splitext(os.path.basename(filedir))[0]
    # vector_filename = base_source + "_gadget_vectors.pkl"


    # if os.path.exists(vector_filename):
    #     df = pandas.read_pickle(vector_filename)
    # else:
    # df.to_pickle(vector_filename)






if __name__ == "__main__":
    main()  # 训练集和测试集分开开开开开
