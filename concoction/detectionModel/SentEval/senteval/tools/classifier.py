

from __future__ import absolute_import, division, unicode_literals

import numpy as np
import copy,datetime
from senteval import utils
import logging
import torch
import torchmetrics
from torch import nn
import torch.nn.functional as F
import nni

class PyTorchClassifier(object):
    def __init__(
        self,
        inputdim,
        nclasses,
        l2reg=0.0,
        batch_size=64,
        seed=1111,
        cudaEfficient=False,
    ):
        # fix seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.inputdim = inputdim
        self.nclasses = nclasses
        self.l2reg = l2reg
        self.batch_size = batch_size
        self.cudaEfficient = cudaEfficient

    def prepare_split(self, X, y, validation_data=None, validation_split=None):
        # Preparing validation data
        assert validation_split or validation_data
        if validation_data is not None:
            trainX, trainy = X, y
            devX, devy = validation_data
        else:
            permutation = np.random.permutation(len(X))
            trainidx = permutation[int(validation_split * len(X)) :]
            devidx = permutation[0 : int(validation_split * len(X))]
            trainX, trainy = X[trainidx], y[trainidx]
            devX, devy = X[devidx], y[devidx]

        # device = torch.device('cpu') if self.cudaEfficient else torch.device('cuda')
        device = "cpu"
        trainX = torch.from_numpy(trainX).to(device, dtype=torch.float32)
        trainy = torch.from_numpy(trainy).to(device, dtype=torch.int64)
        devX = torch.from_numpy(devX).to(device, dtype=torch.float32)
        devy = torch.from_numpy(devy).to(device, dtype=torch.int64)

        return trainX, trainy, devX, devy

    def fit(self, X, y, validation_data=None, validation_split=None, early_stop=True):
        global bestacc, bestrecall, bestprecision
        self.nepoch = 0
        bestf1 = -1
        stop_train = False
        early_stop_count = 0
        # Preparing validation data
        trainX, trainy, devX, devy = self.prepare_split(
            X, y, validation_data, validation_split
        )

        # Training
        while not stop_train and self.nepoch <= self.max_epoch:
            self.trainepoch(trainX, trainy, epoch_size=self.epoch_size)
            accuracy, precision, recall, f1 = self.score(devX, devy)
            # nni.report_intermediate_result(accuracy)
            if f1 > bestf1:
                bestf1 = f1
                bestacc = accuracy
                bestrecall = recall
                bestprecision = precision
                bestmodel = copy.deepcopy(self.model)
            elif early_stop:
                if early_stop_count >= self.tenacity:
                    stop_train = True
                early_stop_count += 1
            logging.info(
                "Training for {4} epochs: f1 = {0}, precision = {2},recall = {3},accuracy = {1}".format(
                    f1, accuracy, precision, recall, self.nepoch
                )
            )
        logging.info(
            "Training best: f1 = {0}, precision = {2},recall = {3},accuracy = {1}".format(
                bestf1, bestacc, bestprecision, bestrecall, self.nepoch
            )
        )
        # nni.report_final_result(bestacc)
        self.model = bestmodel


        return bestf1, bestacc, bestprecision, bestrecall
    
    def fit_increament(self, X, y, validation_data=None, validation_split=None, early_stop=True):
        global bestacc, bestrecall, bestprecision
        self.nepoch = 0
        bestf1 = -1
        stop_train = False
        early_stop_count = 0
        # Preparing validation data
        trainX, trainy, devX, devy = self.prepare_split(
            X, y, validation_data, validation_split
        )

        # Training
        while not stop_train and self.nepoch <= self.max_epoch:
            self.trainepoch(trainX, trainy, epoch_size=self.epoch_size)
            accuracy, precision, recall, f1 = self.score(devX, devy)
            # nni.report_intermediate_result(accuracy)
            if f1 > bestf1:
                bestf1 = f1
                bestacc = accuracy
                bestrecall = recall
                bestprecision = precision
                bestmodel = copy.deepcopy(self.model)
            elif early_stop:
                if early_stop_count >= self.tenacity:
                    stop_train = True
                early_stop_count += 1

        self.model = bestmodel
        return bestf1, bestacc, bestprecision, bestrecall



    def trainepoch(self, X, y, epoch_size):
        self.model.train()
        for _ in range(self.nepoch, self.nepoch + epoch_size):
            permutation = np.random.permutation(len(X))
            all_costs = []
            for i in range(0, len(X), self.batch_size):
                # forward
                idx = (
                    torch.from_numpy(permutation[i : i + self.batch_size])
                    .long()
                    .to(X.device)
                )

                Xbatch = X[idx]
                ybatch = y[idx]

                if self.cudaEfficient:
                    Xbatch = Xbatch.cuda()
                    ybatch = ybatch.cuda()
                output = self.model(Xbatch)
                # loss
                loss = self.loss_fn(output, ybatch)
                all_costs.append(loss.data.item())
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                # Update parameters
                self.optimizer.step()
        self.nepoch += epoch_size

    def score(self, devX, devy):
        self.model.eval()
        correct = 0

        test_acc = torchmetrics.Accuracy()
        test_recall = torchmetrics.Recall(average="none", num_classes=2)
        test_precision = torchmetrics.Precision(average="none", num_classes=2)
        if not isinstance(devX, torch.cuda.FloatTensor) or self.cudaEfficient:
            devX = torch.FloatTensor(devX)
            # .cuda()
            devy = torch.LongTensor(devy)
            # .cuda()
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i : i + self.batch_size]
                ybatch = devy[i : i + self.batch_size]
                if self.cudaEfficient:
                    Xbatch = Xbatch
                    # cuda()
                    ybatch = ybatch
                    # cuda()
                output = self.model(Xbatch)
                pred = output.data.max(1)[1]
                correct += pred.long().eq(ybatch.data.long()).sum().item()
                test_acc(pred, ybatch.data.long())
                test_recall(pred, ybatch.data.long())
                test_precision(pred, ybatch.data.long())
            accuracy = 1.0 * correct / len(devX)
        total_acc = test_acc.compute().numpy()
        total_recall1 = test_recall.compute().numpy()
        total_precision1 = test_precision.compute().numpy()
        total_recall = test_recall.compute().numpy()[1]
        total_precision = test_precision.compute().numpy()[1]

        if total_recall + total_precision == 0:
            total_F1 = 0
        else:
            total_F1 = (2 * total_precision * total_recall) / (
                total_precision + total_recall
            )
        test_precision.reset()
        test_acc.reset()
        test_recall.reset()

        return accuracy, total_precision, total_recall, total_F1

    def predict(self, devX):
        self.model.eval()
        if not isinstance(devX, torch.cuda.FloatTensor):
            devX = torch.FloatTensor(devX)
            # .cuda()
        yhat = np.array([])
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i : i + self.batch_size]
                output = self.model(Xbatch)
                yhat = np.append(yhat, output.data.max(1)[1].cpu().numpy())
        yhat = np.vstack(yhat)
        return yhat

    def predict_proba(self, devX):
        self.model.eval()
        probas = []
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i : i + self.batch_size]
                vals = F.softmax(self.model(Xbatch).data.cpu().numpy())
                if not probas:
                    probas = vals
                else:
                    probas = np.concatenate(probas, vals, axis=0)
        return probas

    def predict_output(self,devX):
        self.model.eval()
        if not isinstance(devX, torch.cuda.FloatTensor):
            devX = torch.FloatTensor(devX)
            # .cuda()
        yhat = np.empty((0, 2))
        with torch.no_grad():
            for i in range(0, len(devX), self.batch_size):
                Xbatch = devX[i : i + self.batch_size]
                output = self.model(Xbatch)
                yhat = np.vstack([yhat,output.data.cpu().numpy()])
        # yhat = np.vstack(yhat)
        return yhat



    def compute_task_metrics_cp(self, task_output, batch_labels ) :
        # clacuilate p-value
        true_label = torch.argmax(batch_labels)

        # pvalue
        task_output_cp = torch.zeros([task_output.shape[0], 1])
        for i in range(batch_labels.shape[1]):
            label = np.zeros(batch_labels.shape[1])
            label[i] = 1
            indices_tem = torch.where(torch.eq(batch_labels, torch.tensor(label))[:, 0])
            indices_tem=indices_tem[0]
            # get probability
            tem = torch.gather(torch.tensor(task_output), 0, indices_tem.unsqueeze(-1)).squeeze()
            for j, tem_pro in zip(indices_tem, tem):
                mask=tem > tem_pro
                num=tem[mask]
                # num = torch.gather(tem,0, torch.where(tem > tem_pro)[0].unsqueeze(-1)).squeeze()
                a = j+ 1
                p_tem = torch.reshape(
                    torch.sub(1, torch.div(num.shape[0], tem.shape[0])), (1, 1))
                part1 = task_output_cp[:int(j)]
                part2 = task_output_cp[int(j) + 1:]
                task_output_cp = torch.cat([part1, p_tem, part2], axis=0)
                a = torch.reshape(task_output_cp, (1, -1))
        min = torch.topk(-torch.reshape(task_output_cp, (1, -1)), task_output_cp.shape[0])
        pindex = min[1].numpy()[0][:int(task_output_cp.shape[0]*0.02)]
        # pindex = torch.reshape(torch.where(condition=task_output_cp < value)[:, 0], (-1, 1))

        return pindex

"""
MLP with Pytorch (nhid=0 --> Logistic Regression)
"""


class MLP(PyTorchClassifier):
    def __init__(
        self,
        params,
        inputdim,
        nclasses,
        l2reg=0.0,
        batch_size=64,
        seed=1111,
        cudaEfficient=False,
    ):
        super(self.__class__, self).__init__(
            inputdim, nclasses, l2reg, batch_size, seed, cudaEfficient
        )
        """
        PARAMETERS:
        -nhid:       number of hidden units (0: Logistic Regression)
        -optim:      optimizer ("sgd,lr=0.1", "adam", "rmsprop" ..)
        -tenacity:   how many times dev acc does not increase before stopping
        -epoch_size: each epoch corresponds to epoch_size pass on the train set
        -max_epoch:  max number of epoches
        -dropout:    dropout for MLP
        """

        self.nhid = 0 if "nhid" not in params else params["nhid"]
        self.optim = "adam" if "optim" not in params else params["optim"]
        self.tenacity = 5 if "tenacity" not in params else params["tenacity"]
        self.epoch_size = 10 if "epoch_size" not in params else params["epoch_size"]
        self.max_epoch =80 if "max_epoch" not in params else params["max_epoch"]
        self.dropout = 0.0 if "dropout" not in params else params["dropout"]
        self.batch_size = 64 if "batch_size" not in params else params["batch_size"]

        
        if params["nhid"] == 0:
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, self.nclasses),
            )
            # ).cuda()
        else:
            self.model = nn.Sequential(
                nn.Linear(self.inputdim, params["nhid"]),
                nn.Dropout(p=self.dropout),
                nn.Sigmoid(),
                nn.Linear(params["nhid"], self.nclasses),
            )
            # ).cuda()
        # self.loss_fn = nn.CrossEntropyLoss().cuda()
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn.size_average = False

        optim_fn, optim_params = utils.get_optimizer(self.optim)
        self.optimizer = optim_fn(self.model.parameters(), **optim_params)
        self.optimizer.param_groups[0]["weight_decay"] = self.l2reg
