import argparse
import logging
import os
import time

import pandas as pd
import numpy as np
# from tqdm import tqdm
import random
from operator import itemgetter
import transformers
import torch
from transformers import AutoModel, AutoTokenizer
import Liger_batch

def loadFile(fpath,ratio):
    logging.info("preprocess data........")
    data_path = fpath
    y = []
    gap = " "
    X_feature = {
        "name": {},
        "X_trace": {},
        "X_testcase": {},
        "X_dynamic": {},
        "label": {},
        "label_group": {},
        "function_nums": {}
    }
    X_feature["name"] = []
    X_feature["X_trace"] = []
    X_feature["X_testcase"] = []
    X_feature["X_dynamic"] = []
    X_feature["label"] = []
    X_feature["label_group"] = []
    X_feature["function_nums"] = []

    # bad,good分别处理
    bad = os.path.join(data_path, "bad")
    good = os.path.join(data_path, "good")
    _bad = os.listdir(bad)
    _good = os.listdir(good)
    bad_nums = min(len(_bad), len(_good))

    # 处理bad
    # count1计算到nums
    count = 0
    # bad下有多少文件夹就有多少函数
    for bad_dir_file in _bad:
        file = os.path.join(bad, bad_dir_file)
        if file.endswith(".txt"):
            flag = "none"
            # X_Graph_Single = np.zeros([Graph_length, Graph_length])
            X_trace_Single = []
            X_testcase_single = []
            X_dynamic_single = []
            # print(file_path)
            f = open(file)
            try:
                for line in f:
                    if line == "-----label-----\n":
                        flag = "label"
                        continue
                    if line == "-----code-----\n":
                        flag = "next"
                        continue
                    if line == "-----children-----\n":
                        flag = "next"
                        continue
                    if line == "-----nextToken-----\n":
                        flag = "next"
                        continue
                    if line == "-----computeFrom-----\n":
                        flag = "next"
                        continue
                    if line == "-----guardedBy-----\n":
                        flag = "next"
                        continue
                    if line == "-----guardedByNegation-----\n":
                        flag = "next"
                        continue
                    if line == "-----lastLexicalUse-----\n":
                        flag = "next"
                        continue
                    if line == "-----jump-----\n":
                        flag = "next"
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
                        flag = "next"
                        continue
                    if line == "=======================\n":
                        break
                    if flag == "next":
                        continue
                    if flag == "label":
                        y = line.split()
                        continue
                    if flag == "code":
                        continue
                    if flag == "children":
                        continue
                    if flag == "nextToken":
                        continue
                    if flag == "computeFrom":
                        continue
                    if flag == "guardedBy":
                        continue
                    if flag == "guardedByNegation":
                        continue
                    if flag == "lastLexicalUse":
                        continue
                    if flag == "jump":
                        continue
                    if flag == "ast_node":
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
                logging.info("please delete the file " + file)

            X_feature["name"].append(file)
            X_feature["X_trace"].append(gap.join(X_trace_Single).split())
            X_feature["X_testcase"].append(gap.join(X_testcase_single).split())
            X_feature["X_dynamic"].append(gap.join(X_dynamic_single).split())
            X_feature["label"].append(int(y[0]))
            if count == bad_nums:
                break
            count += 1


    # 处理good
    count = 0
    # bad下有多少文件夹就有多少函数
    for good_dir_file in _good:
        file = os.path.join(good, good_dir_file)
        if file.endswith(".txt"):
            flag = "none"
            # X_Graph_Single = np.zeros([Graph_length, Graph_length])
            X_trace_Single = []
            X_testcase_single = []
            X_dynamic_single = []
            # print(file_path)
            f = open(file)
            try:
                for line in f:
                    if line == "-----label-----\n":
                        flag = "label"
                        continue
                    if line == "-----code-----\n":
                        flag = "next"
                        continue
                    if line == "-----children-----\n":
                        flag = "next"
                        continue
                    if line == "-----nextToken-----\n":
                        flag = "next"
                        continue
                    if line == "-----computeFrom-----\n":
                        flag = "next"
                        continue
                    if line == "-----guardedBy-----\n":
                        flag = "next"
                        continue
                    if line == "-----guardedByNegation-----\n":
                        flag = "next"
                        continue
                    if line == "-----lastLexicalUse-----\n":
                        flag = "next"
                        continue
                    if line == "-----jump-----\n":
                        flag = "next"
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
                        flag = "next"
                        continue
                    if line == "=======================\n":
                        break
                    if flag == "next":
                        continue
                    if flag == "label":
                        y = line.split()
                        continue
                    if flag == "code":
                        continue
                    if flag == "children":
                        continue
                    if flag == "nextToken":
                        continue
                    if flag == "computeFrom":
                        continue
                    if flag == "guardedBy":
                        continue
                    if flag == "guardedByNegation":
                        continue
                    if flag == "lastLexicalUse":
                        continue
                    if flag == "jump":
                        continue
                    if flag == "ast_node":
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
                logging.info("please delete the file " + file)

            X_feature["name"].append(file)
            X_feature["X_trace"].append(gap.join(X_trace_Single).split())
            X_feature["X_testcase"].append(gap.join(X_testcase_single).split())
            X_feature["X_dynamic"].append(gap.join(X_dynamic_single).split())
            X_feature["label"].append(int(y[0]))
            if count == bad_nums:
                break
            count += 1

    index = list(range(len(X_feature["name"])))
    random.Random(1234).shuffle(index)

    train_idx = int(len(X_feature["name"]) * ratio)

    index_train = index[:train_idx]
    index_test = index[train_idx: ]

    train = {
        "name": {},
        "X_trace": {},
        "X_testcase": {},
        "X_dynamic": {},
        "label": {},
    }
    train["name"] = itemgetter(*index_train)(X_feature["name"])
    train["X_trace"] = itemgetter(*index_train)(X_feature["X_trace"])
    train["X_testcase"] = itemgetter(*index_train)(X_feature["X_testcase"])
    train["X_dynamic"] = itemgetter(*index_train)(X_feature["X_dynamic"])
    train["label"] = itemgetter(*index_train)(X_feature["label"])

    test = {
        "name": {},
        "X_trace": {},
        "X_testcase": {},
        "X_dynamic": {},
        "label": {},
    }
    test["name"] = itemgetter(*index_test)(X_feature["name"])
    test["X_trace"] = itemgetter(*index_test)(X_feature["X_trace"])
    test["X_testcase"] = itemgetter(*index_test)(X_feature["X_testcase"])
    test["X_dynamic"] = itemgetter(*index_test)(X_feature["X_dynamic"])
    test["label"] = itemgetter(*index_test)(X_feature["label"])

    return train, test


def embedding(dynamic, max_length=None):
    # Handle rare token encoding issues in the dataset
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dynamic_feature  = dynamic['X_dynamic']
    sentences = [" ".join(s) for s in dynamic_feature]

    # Tokenization
    batch_dynamic = tokenizer.batch_encode_plus(
        sentences,
        return_tensors="pt",
        max_length=100,
        padding="max_length",
        truncation="longest_first",
    )

    # # Move to the correct device
    for k in batch_dynamic:
        batch_dynamic[k] = batch_dynamic[k].to(device)

    # # Get raw embeddings
    with torch.no_grad():
        dynamic_last_output = model(
            **batch_dynamic, output_hidden_states=True, return_dict=True
        ).last_hidden_state

    dynamic_embedding = torch.reshape(
        torch.nn.AdaptiveMaxPool2d((1, 100))(dynamic_last_output), (-1, 100)
    )

    score = torch.rand(len(dynamic['label']), 2)
    label = torch.LongTensor(dynamic['label'])
    label = torch.unsqueeze(label, dim=1)
    # print("label shape:{},label dtype:{}".format(label.size(), label.dtype))
    y = torch.zeros_like(score).scatter_(1, label, torch.ones_like(label, dtype=torch.float32))
    # print("target shape:{},target dtype:{}".format(y.size(), y.dtype))
    length = dynamic['label']

    dynamic_embedding = dynamic_embedding.numpy()
    y = y.numpy()
    length = np.array(length)

    return dynamic_embedding,y,length,len(dynamic_embedding)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    # --train_data_file "/root/code/Data/CWE-416-1"
    parser.add_argument("--train_data_file", default="/root/code/Data/CWE-416-1", type=str, required=True,
                        help="The input training data file (a text file).")
    # parser.add_argument("--output_dir", default=None, type=str, required=True,
    #                     help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--epoch", default=10, type=str, required=False,
                        help="epoch")
    parser.add_argument("--lr", default=0.0001, type=float, required=False,
                        help="learning_rate")
    parser.add_argument("--n_hidden", default=200, type=int, required=False,
                        help="n_hidden")
    parser.add_argument("--vocabulary_size", default=10000, type=int, required=False,
                        help="learning_rate")
    parser.add_argument("--classes", default=2, type=int, required=False,
                        help="classes")
    parser.add_argument("--vector_length", default=100, type=int, required=False,
                        help="num_of_neurons")
    #因为是single，所以这个参数默认为1
    parser.add_argument("--per_function_traces", default=1, type=int, required=False,
                        help="num_of_trace_for_pre_code")
    parser.add_argument("--train_test_ratio", default=0.8, type=float, required=False,
                        help="train_test_ratio")
    args = parser.parse_args()


    train, test = loadFile(args.train_data_file,args.train_test_ratio)
    train_embedding,train_y,train_length,train_nums = embedding(train)
    test_embedding,test_y,test_length,test_nums = embedding(test)
    liger = Liger_batch.StateTraining(train_embedding, train_y, test_embedding, test_y,
                                      args.vector_length,args.n_hidden,args.classes,args.lr,args.epoch,args.vocabulary_size,
                                      args.per_function_traces,test_nums)
    liger.train_evaluate()


if __name__ == "__main__":
    #此代码为不分每个function的traces，也就是默认trace和label = 1:1
    main()