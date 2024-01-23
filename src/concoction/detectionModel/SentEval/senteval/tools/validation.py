
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
        
        print("Training the detection model...")
        regs = (
            [10**t for t in range(-5, -1)]
            if self.usepytorch
            else [2**t for t in range(-2, 4, 1)]
        )
        if self.noreg:
            regs = [1e-9 if self.usepytorch else 1e9]
        scores = []
        for reg in regs:

            if self.usepytorch:
                clf = MLP(
                    self.classifier_config,
                    inputdim=self.featdim,
                    nclasses=self.nclasses,
                    l2reg=reg,
                    seed=self.seed,
                    cudaEfficient=self.cudaEfficient,
                )


                bestf1, bestacc, bestprecision, bestrecall=clf.fit(
                    self.X["train"],
                    self.y["train"],
                    validation_data=(self.X["valid"], self.y["valid"]),
                )
            else:
                clf = LogisticRegression(C=reg, random_state=self.seed)
                clf.fit(self.X["train"], self.y["train"])


            scores.append(
                np.array(clf.score(self.X["valid"], self.y["valid"]), dtype=float)
            )

        scores_best = -1
        j = -1
        for i in scores:
            j = j + 1
            if i[3] > scores_best:
                scores_best = i[3]
                scores_idx = j

        optreg = regs[scores_idx]
        devaccuracy, devprecision, devrecall, devf1 = scores[scores_idx]

        clf = LogisticRegression(C=optreg, random_state=self.seed)

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
            
           
            path = f"./f1_{bestf1}_{datetime.datetime.now().strftime('%Y-%m-%d')}.h5"
            torch.save(clf, path)

        else:
            clf = LogisticRegression(C=optreg, random_state=self.seed)
            clf.fit(self.X["train"], self.y["train"])

        testaccuracy, testprecision, testrecall, testf1 = clf.score(
            self.X["test"], self.y["test"]
        )

                                                                             #f1 = {testf1:.4f}, precision = {testprecision:.4f}, recall = {testrecall:.4f}, accuracy = {testaccuracy:.4f}
        print(f"Training finished and saving the best trained model, The best validation performance is :\nf1 = {testf1:.4f}, precision = {testprecision:.4f}, recall = {testrecall:.4f}, accuracy = {testaccuracy:.4f}")

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


    def predict(self):
        clf=torch.load(self.modelpath)
       
        yhat=clf.predict(self.X["train"])
        accuracy, precision, recall, f1 = clf.score(self.X["train"], self.y["train"])
        accuracy=round(accuracy,4)
        precision=round(precision,4)
        recall=round(recall,4)
        f1=round(f1,4)
       
        print(f"The test performance is :\nf1 = {f1:.4f}, precision = {precision:.4f},recall = {recall:.4f},accuracy = {accuracy:.4f}")
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
            
        def result_log(acc,precision,recall,f1):
            file_path="./result.log"
            with open(file_path, 'w') as f:
                f.writelines(f"acc,{acc}\n")
                f.writelines(f"pre,{precision}\n")
                f.writelines(f"recall,{recall}\n")
                f.writelines(f"f1,{f1}")
                
        result_log(accuracy,precision,recall,f1)
        
        import pickle
        def result_pkl(acc,precision,recall,f1):
            file_name="result.pkl"
            result_dict = {
                'Accuracy': "{:.4f}".format(acc),
                'Precision': "{:.4f}".format(precision),
                'Recall': "{:.4f}".format(recall),
                'F1 Score': "{:.4f}".format(f1)
            }
            with open(file_name, 'wb') as file:
                pickle.dump(result_dict, file)
        result_pkl(accuracy,precision,recall,f1)
        




