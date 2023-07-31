import os
import shutil

Path = "/../data/Tem/BUG"
from tqdm import tqdm as tqdm
import numpy as np

import logging
import random
from operator import itemgetter

# def loadFile(self, fpath):


def findAllFile(dir):
    for root, ds, fs in os.walk(dir):
        for f in fs:
            yield root, f


data_path = Path
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

# for root, file in tqdm(findAllFile(data_path), desc='dirs'):
for root, file in tqdm(findAllFile(data_path), desc="dirs"):
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

        # Sentence= gap.join(X_dynamic_single).split()
        S_write = " ".join(X_dynamic_single) + "\n"

        with open("output.txt", "a") as f:
            f.write(S_write)

        f.close()

logging.info("Saving success")
