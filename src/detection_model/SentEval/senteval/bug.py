"""
SST - binary classification
"""

from __future__ import absolute_import, division, unicode_literals
import torch
import os
import io
import logging
import numpy as np
import os

# load data text from dir
from tqdm import tqdm
from senteval.tools.validation import SplitClassifier
import random
from operator import itemgetter


class BUGEval(object):
    def __init__(self, modelpath, mode, task_path, nclasses=2, seed=1111):
        self.seed = seed

        # binary of fine-grained
        assert nclasses in [2, 5]
        self.nclasses = nclasses
        self.task_name = "Binary" if self.nclasses == 2 else "Fine-Grained"
        logging.debug(
            "***** Transfer task : BUG %s classification *****\n\n", self.task_name
        )
        if mode == "pred":
            data = self.loadFile_pred(task_path)
            self.sst_data = {"data": data}
        else:
            train, dev, test = self.loadFile(task_path)
            self.sst_data = {"train": train, "dev": dev, "test": test}
        self.modelpath = modelpath
        self.mode = mode

    def loadFile(self, fpath):
        if os.path.exists(os.path.join(fpath, "feature.npy")):
            logging.info("Load data from exist npy")
            X_feature = np.load(
                os.path.join(fpath, "feature.npy"), allow_pickle=True
            ).item()

            index = list(range(len(X_feature["name"])))
            random.Random(123456).shuffle(index)

            train_idx = int(len(X_feature["name"]) * 0.6)
            dev_idx = int(len(X_feature["name"]) * 0.8)
            print("total train files is ", train_idx)
            index_train = index[:train_idx]
            index_dev = index[train_idx:dev_idx]
            index_test = index[dev_idx:]

            train = {
                "name": {},
                "X_Code": {},
                "X_trace": {},
                "X_testcase": {},
                "X_Graph": {},
                "X_Node": {},
                "X_dynamic": {},
                "label": {},
            }
            train["name"] = itemgetter(*index_train)(X_feature["name"])
            train["X_Code"] = itemgetter(*index_train)(X_feature["X_Code"])
            train["X_trace"] = itemgetter(*index_train)(X_feature["X_trace"])
            train["X_testcase"] = itemgetter(*index_train)(X_feature["X_testcase"])
            train["X_Graph"] = itemgetter(*index_train)(X_feature["X_Graph"])
            train["X_Node"] = itemgetter(*index_train)(X_feature["X_Node"])
            train["X_dynamic"] = itemgetter(*index_train)(X_feature["X_dynamic"])
            train["label"] = itemgetter(*index_train)(X_feature["label"])

            dev = {
                "name": {},
                "X_Code": {},
                "X_trace": {},
                "X_testcase": {},
                "X_Graph": {},
                "X_Node": {},
                "X_dynamic": {},
                "label": {},
            }
            dev["name"] = itemgetter(*index_dev)(X_feature["name"])
            dev["X_Code"] = itemgetter(*index_dev)(X_feature["X_Code"])
            dev["X_trace"] = itemgetter(*index_dev)(X_feature["X_trace"])
            dev["X_testcase"] = itemgetter(*index_dev)(X_feature["X_testcase"])
            dev["X_Graph"] = itemgetter(*index_dev)(X_feature["X_Graph"])
            dev["X_Node"] = itemgetter(*index_dev)(X_feature["X_Node"])
            dev["X_dynamic"] = itemgetter(*index_dev)(X_feature["X_dynamic"])
            dev["label"] = itemgetter(*index_dev)(X_feature["label"])

            test = {
                "name": {},
                "X_Code": {},
                "X_trace": {},
                "X_testcase": {},
                "X_Graph": {},
                "X_Node": {},
                "X_dynamic": {},
                "label": {},
            }
            test["name"] = itemgetter(*index_test)(X_feature["name"])
            test["X_Code"] = itemgetter(*index_test)(X_feature["X_Code"])
            test["X_trace"] = itemgetter(*index_test)(X_feature["X_trace"])
            test["X_testcase"] = itemgetter(*index_test)(X_feature["X_testcase"])
            test["X_Graph"] = itemgetter(*index_test)(X_feature["X_Graph"])
            test["X_Node"] = itemgetter(*index_test)(X_feature["X_Node"])
            test["X_dynamic"] = itemgetter(*index_test)(X_feature["X_dynamic"])
            test["label"] = itemgetter(*index_test)(X_feature["label"])

            return train, dev, test
        else:
            logging.info("preprocess data........")

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
                        logging.info("please delete the file " + file)

                    X_feature["name"].append(file_path)
                    X_feature["X_Code"].append(gap.join(X_Code_Single))
                    X_feature["X_trace"].append(gap.join(X_trace_Single).split())
                    X_feature["X_testcase"].append(gap.join(X_testcase_single).split())
                    X_feature["X_Graph"].append(X_Graph_Single)
                    X_feature["X_Node"].append(gap.join(X_Node_Singe))
                    X_feature["X_dynamic"].append(gap.join(X_dynamic_single).split())
                    X_feature["label"].append(int(y[0]))

            index = list(range(len(X_feature["name"])))
            random.Random(123456).shuffle(index)

            train_idx = int(len(X_feature["name"]) * 0.6)
            dev_idx = int(len(X_feature["name"]) * 0.8)

            index_train = index[0:train_idx]
            index_dev = index[train_idx:dev_idx]
            index_test = index[dev_idx + 2 : -1]

            logging.info("Saving embedding...")
            np.save(os.path.join(fpath, "feature.npy"), X_feature)
            logging.info("Saving success")

            train = {
                "name": {},
                "X_Code": {},
                "X_trace": {},
                "X_testcase": {},
                "X_Graph": {},
                "X_Node": {},
                "X_dynamic": {},
                "label": {},
            }
            train["name"] = itemgetter(*index_train)(X_feature["name"])
            train["X_Code"] = itemgetter(*index_train)(X_feature["X_Code"])
            train["X_trace"] = itemgetter(*index_train)(X_feature["X_trace"])
            train["X_testcase"] = itemgetter(*index_train)(X_feature["X_testcase"])
            train["X_Graph"] = itemgetter(*index_train)(X_feature["X_Graph"])
            train["X_Node"] = itemgetter(*index_train)(X_feature["X_Node"])
            train["X_dynamic"] = itemgetter(*index_train)(X_feature["X_dynamic"])
            train["label"] = itemgetter(*index_train)(X_feature["label"])

            dev = {
                "name": {},
                "X_Code": {},
                "X_trace": {},
                "X_testcase": {},
                "X_Graph": {},
                "X_Node": {},
                "X_dynamic": {},
                "label": {},
            }
            dev["name"] = itemgetter(*index_dev)(X_feature["name"])
            dev["X_Code"] = itemgetter(*index_dev)(X_feature["X_Code"])
            dev["X_trace"] = itemgetter(*index_dev)(X_feature["X_trace"])
            dev["X_testcase"] = itemgetter(*index_dev)(X_feature["X_testcase"])
            dev["X_Graph"] = itemgetter(*index_dev)(X_feature["X_Graph"])
            dev["X_Node"] = itemgetter(*index_dev)(X_feature["X_Node"])
            dev["X_dynamic"] = itemgetter(*index_dev)(X_feature["X_dynamic"])
            dev["label"] = itemgetter(*index_dev)(X_feature["label"])

            test = {
                "name": {},
                "X_Code": {},
                "X_trace": {},
                "X_testcase": {},
                "X_Graph": {},
                "X_Node": {},
                "X_dynamic": {},
                "label": {},
            }
            test["name"] = itemgetter(*index_test)(X_feature["name"])
            test["X_Code"] = itemgetter(*index_test)(X_feature["X_Code"])
            test["X_trace"] = itemgetter(*index_test)(X_feature["X_trace"])
            test["X_testcase"] = itemgetter(*index_test)(X_feature["X_testcase"])
            test["X_Graph"] = itemgetter(*index_test)(X_feature["X_Graph"])
            test["X_Node"] = itemgetter(*index_test)(X_feature["X_Node"])
            test["X_dynamic"] = itemgetter(*index_test)(X_feature["X_dynamic"])
            test["label"] = itemgetter(*index_test)(X_feature["label"])

            return train, dev, test

    def loadFile_pred(self, fpath):
        if not os.path.exists(os.path.join(fpath, "feature.npy")):
            logging.info("preprocess data........")

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
                        logging.info("please delete the file " + file)

                    X_feature["name"].append(file_path)
                    X_feature["X_Code"].append(gap.join(X_Code_Single))
                    X_feature["X_trace"].append(gap.join(X_trace_Single).split())
                    X_feature["X_testcase"].append(gap.join(X_testcase_single).split())
                    X_feature["X_Graph"].append(X_Graph_Single)
                    X_feature["X_Node"].append(gap.join(X_Node_Singe))
                    X_feature["X_dynamic"].append(gap.join(X_dynamic_single).split())
                    X_feature["label"].append(int(y[0]))

            logging.info("Saving embedding...")
            np.save(os.path.join(fpath, "feature.npy"), X_feature)
            logging.info("Saving success")

        logging.info("Load data from exist npy")
        X_feature = np.load(
            os.path.join(fpath, "feature.npy"), allow_pickle=True
        ).item()

        # data = {
        #     "name": {},
        #     "X_Code": {},
        #     "X_trace": {},
        #     "X_testcase": {},
        #     "X_Graph": {},
        #     "X_Node": {},
        #     "X_dynamic": {},
        #     "label": {},
        # }
        data = X_feature

        return data

    def run(self, params, batcher):
        sst_embed = {"train": {}, "dev": {}, "test": {}}
        bsize = params.batch_size

        for key in self.sst_data:
            logging.info("Computing embedding for {0}".format(key))
            # Sort to reduce padding
            sorted_data = sorted(
                zip(
                    self.sst_data[key]["name"],
                    self.sst_data[key]["X_Code"],
                    self.sst_data[key]["X_trace"],
                    self.sst_data[key]["X_testcase"],
                    self.sst_data[key]["X_Graph"],
                    self.sst_data[key]["X_Node"],
                    self.sst_data[key]["X_dynamic"],
                    self.sst_data[key]["label"],
                ),
                key=lambda z: (len(z[0]), z[-1]),
            )

            (
                self.sst_data[key]["name"],
                self.sst_data[key]["X_Code"],
                self.sst_data[key]["X_trace"],
                self.sst_data[key]["X_testcase"],
                self.sst_data[key]["X_Graph"],
                self.sst_data[key]["X_Node"],
                self.sst_data[key]["X_dynamic"],
                self.sst_data[key]["label"],
            ) = map(list, zip(*sorted_data))

            sst_embed[key]["X"] = []
            for ii in tqdm(range(0, len(self.sst_data[key]["label"]), bsize)):
                embeddings = []
                batch = (
                    self.sst_data[key]["X_dynamic"][ii : ii + bsize],
                    self.sst_data[key]["X_Node"][ii : ii + bsize],
                    self.sst_data[key]["X_Graph"][ii : ii + bsize],
                )
                embeddings = batcher(params, batch)
                sst_embed[key]["X"].append(embeddings)

            sst_embed[key]["X"] = torch.cat(sst_embed[key]["X"]).detach().numpy()
            sst_embed[key]["y"] = np.array(self.sst_data[key]["label"])
            logging.info("Computed {0} embeddings".format(key))

        config_classifier = {
            "nclasses": self.nclasses,
            "seed": self.seed,
            "usepytorch": params.usepytorch,
            "classifier": params.classifier,
            "modelpath": self.modelpath,
        }

        clf = SplitClassifier(
            X={
                "train": sst_embed["train"]["X"],
                "valid": sst_embed["dev"]["X"],
                "test": sst_embed["test"]["X"],
            },
            y={
                "train": sst_embed["train"]["y"],
                "valid": sst_embed["dev"]["y"],
                "test": sst_embed["test"]["y"],
            },
            name={
                "train": self.sst_data["train"]["name"],
                "valid": self.sst_data["dev"]["name"],
                "test": self.sst_data["test"]["name"],
            },
            config=config_classifier,
        )
        # 预测模式
        if self.mode == "pred":
            clf.predict()
            # clf.predict_increament()
            return
        # 训练模式
        (
            devaccuracy,
            devprecision,
            devrecall,
            devf1,
            testaccuracy,
            testprecision,
            testrecall,
            testf1,
        ) = clf.run()
        # logging.debug('\nDev acc : {0} Test acc : {1} for \
        #     SST {2} classification\n'.format(devaccuracy, testaccuracy, self.task_name))

        return {"devacc": devaccuracy, "testacc": testaccuracy}

    def run_pred(self, params, batcher):
        bsize = params.batch_size
        sst_embed = {"data": {}}
        sst_embed["data"]["X"] = []
        for ii in tqdm(range(0, len(self.sst_data["data"]["label"]), bsize)):
            embeddings = []
            batch = (
                self.sst_data["data"]["X_dynamic"][ii : ii + bsize],
                self.sst_data["data"]["X_Node"][ii : ii + bsize],
                self.sst_data["data"]["X_Graph"][ii : ii + bsize],
            )
            embeddings = batcher(params, batch)
            sst_embed["data"]["X"].append(embeddings)

        sst_embed["data"]["X"] = torch.cat(sst_embed["data"]["X"]).detach().numpy()
        sst_embed["data"]["y"] = np.array(self.sst_data["data"]["label"])
        sst_embed["data"]["name"] = np.array(self.sst_data["data"]["name"])
        logging.info("Computed {0} embeddings".format("data"))

        config_classifier = {
            "nclasses": self.nclasses,
            "seed": self.seed,
            "usepytorch": params.usepytorch,
            "classifier": params.classifier,
            "modelpath": self.modelpath,
        }

        clf = SplitClassifier(
            X={
                "train": sst_embed["data"]["X"],
                "valid": sst_embed["data"]["X"],
                "test": sst_embed["data"]["X"],
            },
            y={
                "train": sst_embed["data"]["y"],
                "valid": sst_embed["data"]["y"],
                "test": sst_embed["data"]["y"],
            },
            name={
                "train": sst_embed["data"]["name"],
                "valid": sst_embed["data"]["name"],
                "test": sst_embed["data"]["name"],
            },
            config=config_classifier,
        )
        # 预测模式
        if self.mode == "pred":
            # clf.predict()
            clf.predict_increament()
            return
        # 训练模式
        (
            devaccuracy,
            devprecision,
            devrecall,
            devf1,
            testaccuracy,
            testprecision,
            testrecall,
            testf1,
        ) = clf.run()
        # logging.debug('\nDev acc : {0} Test acc : {1} for \
        #     SST {2} classification\n'.format(devaccuracy, testaccuracy, self.task_name))

        return {"devacc": devaccuracy, "testacc": testaccuracy}
