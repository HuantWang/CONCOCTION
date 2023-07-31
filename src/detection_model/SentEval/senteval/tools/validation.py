# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Validation and classification
(train)            :  inner-kfold classifier
(train, test)      :  kfold classifier
(train, dev, test) :  split classifier

"""
from __future__ import absolute_import, division, unicode_literals

import datetime

import nni, torch
import logging
import numpy as np
from senteval.tools.classifier import MLP

import sklearn

assert sklearn.__version__ >= "0.18.0", "need to update sklearn to version >= 0.18.0"
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def get_classif_name(classifier_config, usepytorch):
    if not usepytorch:
        modelname = "sklearn-LogReg"
    else:
        nhid = classifier_config["nhid"]
        optim = (
            "adam" if "optim" not in classifier_config else classifier_config["optim"]
        )
        bs = (
            64
            if "batch_size" not in classifier_config
            else classifier_config["batch_size"]
        )
        modelname = "pytorch-MLP-nhid%s-%s-bs%s" % (nhid, optim, bs)
    return modelname


# Pytorch version
class InnerKFoldClassifier(object):
    """
    (train) split classifier : InnerKfold.
    """

    def __init__(self, X, y, config):
        self.X = X
        self.y = y
        self.featdim = X.shape[1]
        self.nclasses = config["nclasses"]
        self.seed = config["seed"]
        self.devresults = []
        self.testresults = []
        self.usepytorch = config["usepytorch"]
        self.classifier_config = config["classifier"]
        self.modelname = get_classif_name(self.classifier_config, self.usepytorch)

        self.k = 5 if "kfold" not in config else config["kfold"]

    def run(self):
        logging.info(
            "Training {0} with (inner) {1}-fold cross-validation".format(
                self.modelname, self.k
            )
        )

        regs = (
            [10**t for t in range(-5, -1)]
            if self.usepytorch
            else [2**t for t in range(-2, 4, 1)]
        )
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=1111)
        innerskf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=1111)
        count = 0
        for train_idx, test_idx in skf.split(self.X, self.y):
            count += 1
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            scores = []
            for reg in regs:
                regscores = []
                for inner_train_idx, inner_test_idx in innerskf.split(X_train, y_train):
                    X_in_train, X_in_test = (
                        X_train[inner_train_idx],
                        X_train[inner_test_idx],
                    )
                    y_in_train, y_in_test = (
                        y_train[inner_train_idx],
                        y_train[inner_test_idx],
                    )
                    if self.usepytorch:
                        clf = MLP(
                            self.classifier_config,
                            inputdim=self.featdim,
                            nclasses=self.nclasses,
                            l2reg=reg,
                            seed=self.seed,
                        )
                        clf.fit(
                            X_in_train,
                            y_in_train,
                            validation_data=(X_in_test, y_in_test),
                        )
                    else:
                        clf = LogisticRegression(C=reg, random_state=self.seed)
                        clf.fit(X_in_train, y_in_train)
                    regscores.append(clf.score(X_in_test, y_in_test))
                scores.append(100 * np.mean(regscores))
            optreg = regs[np.argmax(scores)]
            logging.info(
                "Best param found at split {0}: l2reg = {1} \
                with score {2}".format(
                    count, optreg, np.max(scores)
                )
            )
            self.devresults.append(np.max(scores))

            if self.usepytorch:
                clf = MLP(
                    self.classifier_config,
                    inputdim=self.featdim,
                    nclasses=self.nclasses,
                    l2reg=optreg,
                    seed=self.seed,
                )

                clf.fit(X_train, y_train, validation_split=0.05)
            else:
                clf = LogisticRegression(C=optreg, random_state=self.seed)
                clf.fit(X_train, y_train)

            self.testresults.append(100 * clf.score(X_test, y_test))

        devaccuracy = np.mean(self.devresults)
        testaccuracy = np.mean(self.testresults)
        return devaccuracy, testaccuracy


# class KFoldClassifier(object):
#     """
#     (train, test) split classifier : cross-validation on train.
#     """
#
#     def __init__(self, train, test, config):
#         self.train = train
#         self.test = test
#         self.featdim = self.train["X"].shape[1]
#         self.nclasses = config["nclasses"]
#         self.seed = config["seed"]
#         self.usepytorch = config["usepytorch"]
#         self.classifier_config = config["classifier"]
#         self.modelname = get_classif_name(self.classifier_config, self.usepytorch)
#
#         self.k = 5 if "kfold" not in config else config["kfold"]
#
#     def run(self):
#         # cross-validation
#         logging.info(
#             "Training {0} with {1}-fold cross-validation".format(self.modelname, self.k)
#         )
#         regs = (
#             [10**t for t in range(-5, -1)]
#             if self.usepytorch
#             else [2**t for t in range(-1, 6, 1)]
#         )
#         skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.seed)
#         scores = []
#
#         for reg in regs:
#             scanscores = []
#             for train_idx, test_idx in skf.split(self.train["X"], self.train["y"]):
#                 # Split data
#                 X_train, y_train = (
#                     self.train["X"][train_idx],
#                     self.train["y"][train_idx],
#                 )
#
#                 X_test, y_test = self.train["X"][test_idx], self.train["y"][test_idx]
#
#                 # Train classifier
#                 if self.usepytorch:
#                     clf = MLP(
#                         self.classifier_config,
#                         inputdim=self.featdim,
#                         nclasses=self.nclasses,
#                         l2reg=reg,
#                         seed=self.seed,
#                     )
#                     clf.fit(X_train, y_train, validation_data=(X_test, y_test))
#                 else:
#                     clf = LogisticRegression(C=reg, random_state=self.seed)
#                     clf.fit(X_train, y_train)
#                 score = clf.score(X_test, y_test)
#                 scanscores.append(score)
#             # Append mean score
#             scores.append(100 * np.mean(scanscores))
#
#         # evaluation
#         logging.info(
#             [("reg:" + str(regs[idx]), scores[idx]) for idx in range(len(scores))]
#         )
#         optreg = regs[np.argmax(scores)]
#         devaccuracy = np.max(scores)
#         logging.info(
#             "Cross-validation : best param found is reg = {0} \
#             with score {1}".format(
#                 optreg, devaccuracy
#             )
#         )
#
#         logging.info("Evaluating...")
#         if self.usepytorch:
#             clf = MLP(
#                 self.classifier_config,
#                 inputdim=self.featdim,
#                 nclasses=self.nclasses,
#                 l2reg=optreg,
#                 seed=self.seed,
#             )
#             clf.fit(self.train["X"], self.train["y"], validation_split=0.05)
#         else:
#             clf = LogisticRegression(C=optreg, random_state=self.seed)
#             clf.fit(self.train["X"], self.train["y"])
#         yhat = clf.predict(self.test["X"])
#
#         testaccuracy = clf.score(self.test["X"], self.test["y"])
#         testaccuracy = 100 * testaccuracy, 2
#
#         return devaccuracy, testaccuracy, yhat


class SplitClassifier(object):
    """
    (train, valid, test) split classifier.
    """

    def __init__(self, X, y, name, config):
        self.X = X
        self.y = y
        self.name = name
        self.nclasses = config["nclasses"]
        self.featdim = self.X["train"].shape[1]
        self.seed = config["seed"]
        self.usepytorch = config["usepytorch"]
        self.classifier_config = config["classifier"]
        self.cudaEfficient = (
            False if "cudaEfficient" not in config else config["cudaEfficient"]
        )
        self.modelname = get_classif_name(self.classifier_config, self.usepytorch)
        self.noreg = False if "noreg" not in config else config["noreg"]
        self.config = config
        self.modelpath = config["modelpath"]

    def run(self):
        logging.info("Training {0} with standard validation..".format(self.modelname))
        regs = (
            [10**t for t in range(-5, -1)]
            if self.usepytorch
            else [2**t for t in range(-2, 4, 1)]
        )
        if self.noreg:
            regs = [1e-9 if self.usepytorch else 1e9]
        scores = []
        for reg in regs:
            # logging.info("exist parameter is ", reg)
            if self.usepytorch:
                clf = MLP(
                    self.classifier_config,
                    inputdim=self.featdim,
                    nclasses=self.nclasses,
                    l2reg=reg,
                    seed=self.seed,
                    cudaEfficient=self.cudaEfficient,
                )

                # TODO: Find a hack for reducing nb epoches in SNLI
                clf.fit(
                    self.X["train"],
                    self.y["train"],
                    validation_data=(self.X["valid"], self.y["valid"]),
                )
            else:
                clf = LogisticRegression(C=reg, random_state=self.seed)
                clf.fit(self.X["train"], self.y["train"])

            # a=clf.score(self.X['valid'],self.y['valid'])
            scores.append(
                np.array(clf.score(self.X["valid"], self.y["valid"]), dtype=float)
            )
        logging.info(
            [("reg:" + str(regs[idx]), scores[idx]) for idx in range(len(scores))]
        )
        # a=np.argmax(scores)
        # optreg = regs[np.argmax(scores)]

        scores_best = -1
        j = -1
        for i in scores:
            j = j + 1
            if i[3] > scores_best:
                scores_best = i[3]
                scores_idx = j

        optreg = regs[scores_idx]
        devaccuracy, devprecision, devrecall, devf1 = scores[scores_idx]
        logging.info(
            "Validation : f1 = {1}, precision = {2},recall = {3},accuracy = {4}".format(
                optreg, devf1, devprecision, devrecall, devaccuracy
            )
        )
        clf = LogisticRegression(C=optreg, random_state=self.seed)
        logging.info("Evaluating...")
        if self.usepytorch:
            clf = MLP(
                self.classifier_config,
                inputdim=self.featdim,
                nclasses=self.nclasses,
                l2reg=optreg,
                seed=self.seed,
                cudaEfficient=self.cudaEfficient,
            )

            # TODO: Find a hack for reducing nb epoches in SNLI
            bestf1, _, _, _ = clf.fit(
                self.X["train"],
                self.y["train"],
                validation_data=(self.X["valid"], self.y["valid"]),
            )

            path = f"./f1_{bestf1}_{datetime.datetime.now().strftime('%Y-%m-%d')}.h5"
            torch.save(clf, path)
            logging.info("model clf saved to{0}...".format(path))
        else:
            clf = LogisticRegression(C=optreg, random_state=self.seed)
            clf.fit(self.X["train"], self.y["train"])

        testaccuracy, testprecision, testrecall, testf1 = clf.score(
            self.X["test"], self.y["test"]
        )
        # nni.report_final_result(testaccuracy)
        # testaccuracy = 100*testaccuracy, 2
        logging.info(
            "Test : f1 = {1}, precision = {2},recall = {3},accuracy = {4}".format(
                # optreg, devf1, devprecision, devrecall, devaccuracy
                optreg,
                testf1,
                testprecision,
                testrecall,
                testaccuracy,
            )
        )
        return (
            devaccuracy,
            devprecision,
            devrecall,
            devf1,
            testaccuracy,
            testprecision,
            testrecall,
            testf1,
        )

    # 对指定路径下的特征进行预测输出预测结果以及pos file的地址
    def predict(self):
        clf = torch.load(self.modelpath)
        logging.info("model load from{0}".format(self.modelpath))
        yhat = clf.predict(self.X["train"])
        accuracy, precision, recall, f1 = clf.score(self.X["train"], self.y["train"])
        logging.info(
            "prediction : f1 = {0}, precision = {2},recall = {3},accuracy = {1}".format(
                f1, accuracy, precision, recall
            )
        )
        pos = 0
        all = 0
        pos_file = []
        for i in yhat:
            all = all + 1
            if i == 1:
                pos = pos + 1
        indexes = [index for index, value in enumerate(yhat) if value == 1.0]
        for i in indexes:
            pos_file.append(self.name["train"][i])

        # print(f"this is prediction result:{yhat}")
        print(f"this is positive file path:{pos_file}")
        print(f"this is prediction result: {pos}/{all} pos/all")
        pos = 0
        all = 0
        for i in self.y["train"]:
            all = all + 1
            if i == 1:
                pos = pos + 1
        print(f"this is real result: {pos}/{all} pos/all")
        #

    # 预测+增量学习
    def predict_increament(self):

        model = torch.load(self.modelpath)

        output = model.predict(self.X["test"])
        print(np.squeeze(output))
        acc, pre, recall, f1 = model.score(self.X["test"], self.y["test"])
        logging.info(
            "predict (before Incremental Learning) : f1 = {0}, precision = {1},recall = {2},accuracy = {3}".format(
                f1, pre, recall, acc
            )
        )

        # 增量学习 对x["test"]预测结果计算置信度,将不信任的结果取出放入X["train"]并从X["test"]中删除,对X["train"]训练后再次对X["test"]预测
        yhat = model.predict_output(self.X["test"])
        y = torch.zeros((self.y["test"].shape[0], 2))
        y[:, 1] = torch.tensor(self.y["test"])
        y[:, 0] = 1 - torch.tensor(self.y["test"])
        index = model.compute_task_metrics_cp(torch.Tensor(yhat), y)
        t = 0
        yhatt = np.argmax(yhat, axis=1)
        for i in range(0, len(yhatt) - 1):
            if self.y["test"][i] != yhatt[i]:
                t = t + 1
                print(
                    "this is file name {0},real label:{1},predict label:{2},index:{3}".format(
                        self.name["test"][i], self.y["test"][i], yhatt[i], i
                    )
                )
        print(f"置信度不准确的个数：{len(index)},实际置信度不准确真的不准确个数：{t}")

        # TODO: 给test集减index feature ratio计算有问题

        self.X["train"] = torch.tensor(self.X["test"][index])
        self.y["train"] = torch.tensor(self.y["test"][index])
        self.X["train"] = self.X["train"].numpy()
        self.y["train"] = self.y["train"].numpy()

        idx = torch.tensor(
            [i for i in range(self.X["test"].shape[0]) if i not in index]
        )
        self.X["test"] = torch.index_select(torch.tensor(self.X["test"]), 0, idx)
        self.y["test"] = torch.index_select(torch.tensor(self.y["test"]), 0, idx)
        self.X["test"] = self.X["test"].numpy()
        self.y["test"] = self.y["test"].numpy()

        ratio = len(index) / self.X["test"].shape[0]
        logging.info(
            "增量学习的数据量为:{0}/{1} {2} ".format(len(index), self.X["test"].shape[0], ratio)
        )

        model.fit(
            self.X["train"],
            self.y["train"],
            validation_data=(self.X["train"], self.y["train"]),
        )

        acc, pre, recall, f1 = model.score(self.X["test"], self.y["test"])
        logging.info(
            "predict (after Incremental Learning) : f1 = {0}, precision = {1},recall = {2},accuracy = {3}".format(
                f1, pre, recall, acc
            )
        )

        print("predict() end....")

        return acc, pre, recall, f1
