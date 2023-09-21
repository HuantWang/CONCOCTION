
from __future__ import absolute_import, division, unicode_literals

import datetime
from sklearn.metrics import confusion_matrix
import nni,torch
import logging,os,re
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



class SplitClassifier(object):
    """
    (train, valid, test) split classifier.
    """

    def __init__(self, X, y, name,config):
        self.X = X
        self.y = y
        self.name=name
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
        self.modelpath=config["modelpath"]

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


            bestf1,_,_,_=clf.fit(
                self.X["train"],
                self.y["train"],
                validation_data=(self.X["valid"], self.y["valid"]),
            )
            
            #  p=os.path.join(os.path.dirname(os.path.realpath(__file__)),"./f1_{bestf1}_{datetime.datetime.now().strftime('%Y-%m-%d')}.h5")
            path = f"./f1_{bestf1}_{datetime.datetime.now().strftime('%Y-%m-%d')}.h5"
            torch.save(clf, path)
            logging.info(
                "model clf saved to{0}...".format(
                    path
                )
            )
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
                optreg,testf1,testprecision,testrecall,testaccuracy
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

#对指定路径下的特征进行预测输出预测结果以及pos file的地址
    def predict(self):
        clf=torch.load(self.modelpath)
        logging.info(
            "model load from{0}".format(
                self.modelpath
            )
        )
        yhat=clf.predict(self.X["train"])
        accuracy, precision, recall, f1 = clf.score(self.X["train"], self.y["train"])
        logging.info(
            "prediction : f1 = {0}, precision = {2},recall = {3},accuracy = {1}".format(
                f1, accuracy, precision, recall
            )
        )
        pos=0
        all=0
        pos_file=[]
        for i in yhat:
            all = all + 1
            if i == 1:
                pos = pos + 1
        indexes = [index for index, value in enumerate(yhat) if value == 1.0]
        for i in indexes:
            pos_file.append(self.name["train"][i])




