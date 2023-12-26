import argparse
import logging
import math
import os,nni
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
# from tqdm import tqdm
import random
from operator import itemgetter
import transformers
import torch
from transformers import AutoModel, AutoTokenizer
import Liger_batch
#python /home/ExperimentalEvaluation/lingerwj/run_group.py  --data_file /home/ExperimentalEvaluation/data/github_0.6_new/test --model_to_load /home/ExperimentalEvaluation/lingerwj/savedmodel/0.7207637231503581.cpkt
#python /home/ExperimentalEvaluation/lingerwj/run_group.py  --data_file /home/ExperimentalEvaluation/data/github_0.6_new/train
def loadFile_pre(fpath,ratio,per_function_traces):
    def findAllFile(dir):
                for root, ds, fs in os.walk(dir):
                    for f in fs:
                        yield root, f
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
    }
    X_feature["name"] = []
    X_feature["X_trace"] = []
    X_feature["X_testcase"] = []
    X_feature["X_dynamic"] = []
    X_feature["label"] = []



    for root, file in tqdm(findAllFile(data_path), desc="dirs"):
        trace=file
        function_dir_path=root
        trace_path = os.path.join(function_dir_path, trace)

        if trace.endswith(".txt"):
            flag = "none"
            X_trace_Single = []
            X_testcase_single = []
            X_dynamic_single = []
            # print(file_path)
            f = open(trace_path)
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
                logging.info("please delete the file " + trace)

            X_feature["name"].append(trace_path)
            X_feature["X_trace"].append(gap.join(X_trace_Single).split())
            X_feature["X_testcase"].append(gap.join(X_testcase_single).split())
            X_feature["X_dynamic"].append(gap.join(X_dynamic_single).split())
            X_feature["label"].append(int(y[0]))
    
    index = list(range(len(X_feature["name"])))
    random.Random(123456).shuffle(index)

    train_idx = int(len(X_feature["name"]))
  

    index_train = index[0:train_idx]


    logging.info("Saving embedding...")
    np.save(os.path.join(fpath, "feature.npy"), X_feature)
    logging.info("Saving success")

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






    
    return train

def loadFile_1(fpath,ratio,per_function_traces):
    def findAllFile(dir):
                for root, ds, fs in os.walk(dir):
                    for f in fs:
                        yield root, f
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
    }
    X_feature["name"] = []
    X_feature["X_trace"] = []
    X_feature["X_testcase"] = []
    X_feature["X_dynamic"] = []
    X_feature["label"] = []



    for root, file in tqdm(findAllFile(data_path), desc="dirs"):
        trace=file
        function_dir_path=root
        trace_path = os.path.join(function_dir_path, trace)

        if trace.endswith(".txt"):
            flag = "none"
            X_trace_Single = []
            X_testcase_single = []
            X_dynamic_single = []
            # print(file_path)
            f = open(trace_path)
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
                logging.info("please delete the file " + trace)

            X_feature["name"].append(trace_path)
            X_feature["X_trace"].append(gap.join(X_trace_Single).split())
            X_feature["X_testcase"].append(gap.join(X_testcase_single).split())
            X_feature["X_dynamic"].append(gap.join(X_dynamic_single).split())
            X_feature["label"].append(int(y[0]))
    
    index = list(range(len(X_feature["name"])))
    random.Random(123456).shuffle(index)

    train_idx = int(len(X_feature["name"]) *ratio)
  

    index_train = index[0:train_idx]
    index_test = index[train_idx  :-1]

    logging.info("Saving embedding...")
    np.save(os.path.join(fpath, "feature.npy"), X_feature)
    logging.info("Saving success")

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

def loadFile(fpath,ratio,per_function_traces):
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
    nums = min(len(_bad), len(_good))
    # 正负样本的训练集和测试集比例,只是正样本或者负样本的数量，也就是train整个数据集的一半
    train_nums = math.floor(nums * ratio)

    # 处理bad
    # count1计算到nums
    count = 0
    # 计算到训练集的比例后traces有多少条
    train_traces_num_bad = 0
    traces_num_bad = 0
    # bad下有多少文件夹就有多少函数
    for bad_dir_file in _bad:
        function_dir_path = os.path.join(bad, bad_dir_file)
        traces = os.listdir(function_dir_path)

        # 处理每个函数包含的每一条路径
        trace_count = 0
        for trace in traces:
            trace_path = os.path.join(function_dir_path, trace)

            if trace.endswith(".txt"):
                flag = "none"
                X_trace_Single = []
                X_testcase_single = []
                X_dynamic_single = []
                # print(file_path)
                f = open(trace_path)
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
                    logging.info("please delete the file " + trace)

                X_feature["name"].append(trace_path)
                X_feature["X_trace"].append(gap.join(X_trace_Single).split())
                X_feature["X_testcase"].append(gap.join(X_testcase_single).split())
                X_feature["X_dynamic"].append(gap.join(X_dynamic_single).split())
                X_feature["label"].append(int(y[0]))
                trace_count += 1
                if trace_count == per_function_traces:
                    break
        #规整化per_function_traces，多退少补
        real_traces = per_function_traces-trace_count
        if real_traces > 0:
            for i in range(real_traces):
                X_feature["name"].append(0)
                X_feature["X_trace"].append(gap.join(list(" ")).split())
                X_feature["X_testcase"].append(gap.join(list(" ")).split())
                X_feature["X_dynamic"].append(gap.join(list(" ")).split())
                X_feature["label"].append(-1)

        X_feature["label_group"].append(1)
        X_feature["function_nums"].append(per_function_traces)
        traces_num_bad = traces_num_bad + per_function_traces
        count = count + 1
        if count <= train_nums:
            train_traces_num_bad = train_traces_num_bad + per_function_traces


        if count == nums:
            break

    # 处理good
    count = 0
    train_traces_num_good = 0
    traces_num_good = 0

    # bad下有多少文件夹就有多少函数
    for good_dir_file in _good:
        function_dir_path = os.path.join(good, good_dir_file)
        traces = os.listdir(function_dir_path)

        # 处理每个函数包含的每一条路径
        trace_count = 0
        for trace in traces:
            trace_path = os.path.join(function_dir_path, trace)

            if trace.endswith(".txt"):
                flag = "none"
                # X_Graph_Single = np.zeros([Graph_length, Graph_length])
                X_trace_Single = []
                X_testcase_single = []
                X_dynamic_single = []
                # print(file_path)
                f = open(trace_path)
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
                    logging.info("please delete the file " + trace)

                X_feature["name"].append(trace_path)
                X_feature["X_trace"].append(gap.join(X_trace_Single).split())
                X_feature["X_testcase"].append(gap.join(X_testcase_single).split())
                X_feature["X_dynamic"].append(gap.join(X_dynamic_single).split())
                X_feature["label"].append(int(y[0]))
                trace_count += 1
                if trace_count == per_function_traces:
                    break
        # 规整化per_function_traces，多退少补
        real_traces = per_function_traces - trace_count
        if real_traces > 0:
            for i in range(real_traces):
                X_feature["name"].append(0)
                X_feature["X_trace"].append(gap.join(list(" ")).split())
                X_feature["X_testcase"].append(gap.join(list(" ")).split())
                X_feature["X_dynamic"].append(gap.join(list(" ")).split())
                X_feature["label"].append(-1)
        X_feature["label_group"].append(0)
        X_feature["function_nums"].append(per_function_traces)
        traces_num_good = traces_num_good + per_function_traces
        count = count + 1
        if count <= train_nums:
            train_traces_num_good = train_traces_num_good + per_function_traces


        if count == nums:
            break

    # index = list(range(len(X_feature["label_group"])))
    # 不能直接打乱，正负样本分别划分

    index_trace = list(range(len(X_feature["X_dynamic"])))
    index_y = list(range(len(X_feature["label_group"])))
    # 先负后正
    index_bad_train_trace = index_trace[:train_traces_num_bad]
    index_bad_train_y = index_y[:train_nums]

    index_bad_test_trace = index_trace[train_traces_num_bad:traces_num_bad]
    index_bad_test_y = index_y[train_nums:nums]

    index_good_train_trace = index_trace[traces_num_bad:traces_num_bad + train_traces_num_good]
    index_good_train_y = index_y[nums:nums + train_nums]

    index_good_test_trace = index_trace[traces_num_bad + train_traces_num_good:]
    index_good_test_y = index_y[nums + train_nums:]

    index_train_trace = index_bad_train_trace + index_good_train_trace
    index_train_y = index_bad_train_y + index_good_train_y
    index_test_trace = index_bad_test_trace + index_good_test_trace
    index_test_y = index_bad_test_y + index_good_test_y

    #不用保存，没有用
    # logging.info("Saving embedding...")
    # np.save(os.path.join(fpath, "feature.npy"), X_feature)
    # logging.info("Saving success")

    train = {
        "name": {},
        "X_trace": {},
        "X_testcase": {},
        "X_dynamic": {},
        "label": {},
        "label_group": {},
        "function_nums": {}
    }
    train["name"] = itemgetter(*index_train_trace)(X_feature["name"])
    train["X_trace"] = itemgetter(*index_train_trace)(X_feature["X_trace"])
    train["X_testcase"] = itemgetter(*index_train_trace)(X_feature["X_testcase"])
    train["X_dynamic"] = itemgetter(*index_train_trace)(X_feature["X_dynamic"])
    train["label"] = itemgetter(*index_train_trace)(X_feature["label"])
    train["label_group"] = itemgetter(*index_train_y)(X_feature["label_group"])
    train["function_nums"] = itemgetter(*index_train_y)(X_feature["function_nums"])

    test = {
        "name": {},
        "X_trace": {},
        "X_testcase": {},
        "X_dynamic": {},
        "label": {},
        "label_group": {},
        "function_nums": {}
    }
    test["name"] = itemgetter(*index_test_trace)(X_feature["name"])
    test["X_trace"] = itemgetter(*index_test_trace)(X_feature["X_trace"])
    test["X_testcase"] = itemgetter(*index_test_trace)(X_feature["X_testcase"])
    test["X_dynamic"] = itemgetter(*index_test_trace)(X_feature["X_dynamic"])
    test["label"] = itemgetter(*index_test_trace)(X_feature["label"])
    test["label_group"] = itemgetter(*index_test_y)(X_feature["label_group"])
    test["function_nums"] = itemgetter(*index_test_y)(X_feature["function_nums"])

    return train, test
def loadFilePredicCVE(fpath,ratio,per_function_traces):
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
    print(f"this is X_feature:{X_feature}")

    # bad,good分别处理
    bad = os.path.join(data_path, "bad")
    good = os.path.join(data_path, "bad")
    _bad = os.listdir(bad)
    _good = _bad
    nums =len(_bad)
    # 正负样本的训练集和测试集比例,只是正样本或者负样本的数量，也就是train整个数据集的一半
    train_nums = math.floor(nums * ratio)

    # 处理bad
    # count1计算到nums
    count = 0
    # 计算到训练集的比例后traces有多少条
    train_traces_num_bad = 0
    traces_num_bad = 0
    # bad下有多少文件夹就有多少函数
    for bad_dir_file in _bad:
        traces=[]
        trace = os.path.join(bad, bad_dir_file)
        traces.append(trace)
        # 处理每个函数包含的每一条路径
        trace_count = 0

        for trace in traces:
            trace_path = os.path.join(bad, trace)
            print(f"trace_path:{trace_path}")


            if trace.endswith(".txt"):
                flag = "none"
                X_trace_Single = []
                X_testcase_single = []
                X_dynamic_single = []
                # print(file_path)
                f = open(trace_path)
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
                    logging.info("please delete the file " + trace)

                X_feature["name"].append(trace_path)
                X_feature["X_trace"].append(gap.join(X_trace_Single).split())
                X_feature["X_testcase"].append(gap.join(X_testcase_single).split())
                X_feature["X_dynamic"].append(gap.join(X_dynamic_single).split())
                X_feature["label"].append(int(y[0]))
                trace_count += 1
                if trace_count == per_function_traces:
                    break

        #规整化per_function_traces，多退少补
        real_traces = per_function_traces-trace_count
        if real_traces > 0:
            for i in range(real_traces):
                X_feature["name"].append(0)
                X_feature["X_trace"].append(gap.join(list(" ")).split())
                X_feature["X_testcase"].append(gap.join(list(" ")).split())
                X_feature["X_dynamic"].append(gap.join(list(" ")).split())
                X_feature["label"].append(-1)

        X_feature["label_group"].append(1)
        X_feature["function_nums"].append(per_function_traces)
        traces_num_bad = traces_num_bad + per_function_traces
        count = count + 1
        if count <= train_nums:
            train_traces_num_bad = train_traces_num_bad + per_function_traces


        if count == nums:
            break

    # 处理good
    count = 0
    train_traces_num_good = 0
    traces_num_good = 0

    # bad下有多少文件夹就有多少函数
    for good_dir_file in _good:
        traces = []
        trace = os.path.join(good, good_dir_file)
        traces.append(trace)
        # 处理每个函数包含的每一条路径
        trace_count = 0


        # 处理每个函数包含的每一条路径
        trace_count = 0
        for trace in traces:
            trace_path = os.path.join(good, trace)

            if trace.endswith(".txt"):
                flag = "none"
                # X_Graph_Single = np.zeros([Graph_length, Graph_length])
                X_trace_Single = []
                X_testcase_single = []
                X_dynamic_single = []
                # print(file_path)
                f = open(trace_path)
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
                    logging.info("please delete the file " + trace)

                X_feature["name"].append(trace_path)
                X_feature["X_trace"].append(gap.join(X_trace_Single).split())
                X_feature["X_testcase"].append(gap.join(X_testcase_single).split())
                X_feature["X_dynamic"].append(gap.join(X_dynamic_single).split())
                X_feature["label"].append(int(y[0]))
                trace_count += 1
                if trace_count == per_function_traces:
                    break
        # 规整化per_function_traces，多退少补
        real_traces = per_function_traces - trace_count
        if real_traces > 0:
            for i in range(real_traces):
                X_feature["name"].append(0)
                X_feature["X_trace"].append(gap.join(list(" ")).split())
                X_feature["X_testcase"].append(gap.join(list(" ")).split())
                X_feature["X_dynamic"].append(gap.join(list(" ")).split())
                X_feature["label"].append(-1)
        X_feature["label_group"].append(1)
        X_feature["function_nums"].append(1)
        traces_num_good = traces_num_good + per_function_traces
        count = count + 1
        if count <= train_nums:
            train_traces_num_good = train_traces_num_good + per_function_traces


        if count == nums:
            break

    # index = list(range(len(X_feature["label_group"])))
    # 不能直接打乱，正负样本分别划分

    index_trace = list(range(len(X_feature["X_dynamic"])))
    index_y = list(range(len(X_feature["label_group"])))
    # 先负后正
    index_bad_train_trace = index_trace[:train_traces_num_bad]
    index_bad_train_y = index_y[:train_nums]

    index_bad_test_trace = index_trace[train_traces_num_bad:traces_num_bad]
    index_bad_test_y = index_y[train_nums:nums]

    index_good_train_trace = index_trace[traces_num_bad:traces_num_bad + train_traces_num_good]
    index_good_train_y = index_y[nums:nums + train_nums]

    index_good_test_trace = index_trace[traces_num_bad + train_traces_num_good:]
    index_good_test_y = index_y[nums + train_nums:]

    index_train_trace = index_bad_train_trace + index_good_train_trace
    index_train_y = index_bad_train_y + index_good_train_y
    index_test_trace = index_bad_test_trace + index_good_test_trace
    index_test_y = index_bad_test_y + index_good_test_y

    #不用保存，没有用
    # logging.info("Saving embedding...")
    # np.save(os.path.join(fpath, "feature.npy"), X_feature)
    # logging.info("Saving success")

    train = {
        "name": {},
        "X_trace": {},
        "X_testcase": {},
        "X_dynamic": {},
        "label": {},
        "label_group": {},
        "function_nums": {}
    }
    train["name"] = itemgetter(*index_train_trace)(X_feature["name"])
    train["X_trace"] = itemgetter(*index_train_trace)(X_feature["X_trace"])
    train["X_testcase"] = itemgetter(*index_train_trace)(X_feature["X_testcase"])
    train["X_dynamic"] = itemgetter(*index_train_trace)(X_feature["X_dynamic"])
    train["label"] = itemgetter(*index_train_trace)(X_feature["label"])
    train["label_group"] = itemgetter(*index_train_y)(X_feature["label_group"])
    train["function_nums"] = itemgetter(*index_train_y)(X_feature["function_nums"])

    test = {
        "name": {},
        "X_trace": {},
        "X_testcase": {},
        "X_dynamic": {},
        "label": {},
        "label_group": {},
        "function_nums": {}
    }
    test["name"] = itemgetter(*index_test_trace)(X_feature["name"])
    test["X_trace"] = itemgetter(*index_test_trace)(X_feature["X_trace"])
    test["X_testcase"] = itemgetter(*index_test_trace)(X_feature["X_testcase"])
    test["X_dynamic"] = itemgetter(*index_test_trace)(X_feature["X_dynamic"])
    test["label"] = itemgetter(*index_test_trace)(X_feature["label"])
    test["label_group"] = itemgetter(*index_test_y)(X_feature["label_group"])
    test["function_nums"] = itemgetter(*index_test_y)(X_feature["function_nums"])

    return train, test
def embedding(dynamic, max_length=None):
    # Handle rare token encoding issues in the dataset
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

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

    score = torch.rand(len(dynamic['label_group']), 2)
    label = torch.LongTensor(dynamic['label_group'])
    label = torch.unsqueeze(label, dim=1)
    # print("label shape:{},label dtype:{}".format(label.size(), label.dtype))
    y = torch.zeros_like(score).scatter_(1, label, torch.ones_like(label, dtype=torch.float32))
    # print("target shape:{},target dtype:{}".format(y.size(), y.dtype))
    length = dynamic['function_nums']
    try:
        dynamic_embedding = dynamic_embedding.numpy()
    except:
        dynamic_embedding = dynamic_embedding.cpu().numpy()
    y = y.numpy()
    length = np.array(length)

    return dynamic_embedding,y,length,len(dynamic_embedding)
def embedding_1(dynamic, max_length=None):
    # Handle rare token encoding issues in the dataset
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dynamic_feature  = dynamic['X_dynamic']
    sentences = [" ".join(s) for s in dynamic_feature]

    # Tokenization
    batch_dynamic = tokenizer.batch_encode_plus(
        sentences,
        return_tensors="pt",
        max_length=64,
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

    def convert(label_tuple):
        label_np = np.array(label_tuple)
        # 将标签转换为 one-hot 编码
        label_onehot = np.eye(2)[label_np]
        label_tensor = torch.from_numpy(label_onehot)
        return label_tensor
    y=convert(dynamic['label'])
    
    length=np.array(dynamic['label'])
    try:
        dynamic_embedding = dynamic_embedding.numpy()
    except:
        dynamic_embedding = dynamic_embedding.cpu().numpy()
    y = y.numpy()


    return dynamic_embedding,y,length,len(dynamic_embedding)



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    # --train_data_file "/root/code/Data/CWE-416-1"
    # /home/ExperimentalEvaluation/data/github_0.6_new/train
    parser.add_argument("--data_file", default="/home/ExperimentalEvaluation/data/github_0.6_new/train", type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--model_to_load", default=None, type=str, required=False,
                        help="mode_to_load_to_prediction")
    parser.add_argument("--epoch", default=10, type=str, required=False,
                        help="epoch")
    parser.add_argument("--lr", default=0.001, type=float, required=False,
                        help="learning_rate")
    parser.add_argument("--n_hidden", default=200, type=int, required=False,
                        help="n_hidden")
    parser.add_argument("--vocabulary_size", default=10000, type=int, required=False,
                        help="learning_rate")
    parser.add_argument("--classes", default=2, type=int, required=False,
                        help="classes")
    parser.add_argument("--vector_length", default=100, type=int, required=False,
                        help="num_of_neurons")
    parser.add_argument("--per_function_traces", default=1, type=int, required=False,
                        help="num_of_trace_for_pre_code")
    parser.add_argument("--train_test_ratio", default=0.8, type=float, required=False,
                        help="train_test_ratio")
    args = parser.parse_args()
    # RECEIVED_PARAMS = nni.get_next_parameter()
    # args = vars(args)
    # args.update(RECEIVED_PARAMS)
    

    # train, test = loadFile(args.train_data_file,args.train_test_ratio,args.per_function_traces)
    train, test = loadFile_1(args.data_file,args.train_test_ratio,args.per_function_traces)
    # print("this is size of train and test:")
    # print(len(train["name"]))
    # print(len(test["name"]))
    # train_embedding,train_y,train_length,train_nums = embedding(train)
    train_embedding,train_y,train_length,train_nums = embedding_1(train)
    test_embedding,test_y,test_length,test_nums = embedding_1(test)
    # test_embedding,test_y,test_length,test_nums = embedding(test)
    print("this is size of train_y and test_y:")
    print(len(train_y))
    print(len(test_y))
    
    if not args.model_to_load==None:
        liger = Liger_batch.StateTraining_pre(train_embedding, train_y, test_embedding, test_y,
                                      args.vector_length,args.n_hidden,args.classes,args.lr,args.epoch,args.vocabulary_size,
                                      args.per_function_traces,test_nums,args.model_to_load)
        liger.train_evaluate()
        return
    
    liger = Liger_batch.StateTraining(train_embedding, train_y, test_embedding, test_y,
                                      args.vector_length,args.n_hidden,args.classes,args.lr,args.epoch,args.vocabulary_size,
                                      args.per_function_traces,test_nums)
    liger.train_evaluate()
def mainPredic():
    parser = argparse.ArgumentParser()

    ## Required parameters
    # --train_data_file "/root/code/Data/CWE-416-1"
    parser.add_argument("--train_data_file", default="/root/lingerwj/Data/badall", type=str, required=True,
                        help="The input training data file (a text file).")
    # parser.add_argument("--output_dir", default=None, type=str, required=True,
    #                     help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--epoch", default=5, type=str, required=False,
                        help="epoch")
    parser.add_argument("--lr", default=0.001, type=float, required=False,
                        help="learning_rate")
    parser.add_argument("--n_hidden", default=200, type=int, required=False,
                        help="n_hidden")
    parser.add_argument("--vocabulary_size", default=10000, type=int, required=False,
                        help="learning_rate")
    parser.add_argument("--classes", default=2, type=int, required=False,
                        help="classes")
    parser.add_argument("--vector_length", default=100, type=int, required=False,
                        help="num_of_neurons")
    parser.add_argument("--per_function_traces", default=2, type=int, required=False,
                        help="num_of_trace_for_pre_code")
    parser.add_argument("--train_test_ratio", default=0.8, type=float, required=False,
                        help="train_test_ratio")
    args = parser.parse_args()


    trainPrectCve = loadFile_pre(args.train_data_file,args.train_test_ratio,args.per_function_traces)
    print("this is size of trainPrectCve :")
    print(trainPrectCve)
    print(len(trainPrectCve["name"]))

    trainPrectCve_embedding,trainPrectCve_y,trainPrectCve_length,trainPrectCve_nums = embedding(trainPrectCve)

    print("this is size of train_y and test_y:")
    print(len(train_y))
    print(len(test_y))

    liger = Liger_batch.StateTrainingPredic(train_embedding, train_y, test_embedding, test_y,
                                      args.vector_length,args.n_hidden,args.classes,args.lr,args.epoch,args.vocabulary_size,
                                      args.per_function_traces,test_nums)
    liger.train_evaluate()

if __name__ == "__main__":
    # 此代码为分每个function的traces，也就是默认function(traces)和label = per_function_traces:1
    # mainPredic()
    main()