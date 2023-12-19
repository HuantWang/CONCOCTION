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

def loadFile(fpath,):
    if os.path.exists(os.path.join(fpath, "feature.npy")):
        logging.info("Load data from exist npy")
        X_feature = np.load(
            os.path.join(fpath, "feature.npy"), allow_pickle=True
        ).item()

        index = list(range(len(X_feature["name"])))
        #不能打乱
        # random.Random(123456).shuffle(index)

        train_idx = int(len(X_feature["name"]) * 0.8)

        index_train = index[:train_idx]
        index_test = index[train_idx:]

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
    else:
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

        files = os.listdir(data_path)
        for file in files:
            file_path = os.path.join(data_path,file)
            if file.endswith(".txt"):
                flag = "none"
                # X_Graph_Single = np.zeros([Graph_length, Graph_length])
                X_trace_Single = []
                X_testcase_single = []
                X_dynamic_single = []
                # print(file_path)
                f = open(file_path)
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

                X_feature["name"].append(file_path)
                X_feature["X_trace"].append(gap.join(X_trace_Single).split())
                X_feature["X_testcase"].append(gap.join(X_testcase_single).split())
                X_feature["X_dynamic"].append(gap.join(X_dynamic_single).split())
                X_feature["label"].append(int(y[0]))

        index = list(range(len(X_feature["name"])))
        # 不能打乱
        # random.Random(123456).shuffle(index)

        train_idx = int(len(X_feature["name"]) * 0.8)

        index_train = index[:train_idx]
        index_test = index[train_idx:]

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


def embedding(dynamic, max_length=None):
    # Handle rare token encoding issues in the dataset
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dynamic  = dynamic['X_dynamic']
    sentences = [" ".join(s) for s in dynamic]

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

    embeddings = torch.reshape(
        torch.nn.AdaptiveMaxPool2d((1, 100))(dynamic_last_output), (-1, 100)
    )

    return embeddings


def main():
    # parser = argparse.ArgumentParser()
    #
    # ## Required parameters
    # parser.add_argument("--train_data_file", default=None, type=str, required=True,
    #                     help="The input training data file (a text file).")
    # parser.add_argument("--output_dir", default=None, type=str, required=True,
    #                     help="The output directory where the model predictions and checkpoints will be written.")
    # args = parser.parse_args()
    #
    # name = os.path.basename(args.train_data_file) + "-" + str(round(time.time()))
    # base = os.path.join("result", name)
    # vector_filename = base + "_vectors.pkl"
    # # vector_filename = "result/CWE-200-1649648216_gadget_vectors.pkl"
    # vector_length = 100
    # if os.path.exists(vector_filename):
    #     df = pd.read_pickle(vector_filename)
    # else:
    #     df = get_vectors_df(args.train_data_file, vector_length)
    #     df.to_pickle(vector_filename)
    train, test = loadFile("/root/code/Data/CWE-416")
    embedding_train = embedding(train)
    embedding_test = embedding(test)
    print("")




if __name__ == "__main__":
    main()