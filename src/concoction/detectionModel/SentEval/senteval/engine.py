
from __future__ import absolute_import, division, unicode_literals

import string

from senteval.bug import BUGEval
from senteval import utils
import random


class SE(object):
    def __init__(self, params, batcher, prepare=None):
        # parameters
        params = utils.dotdict(params)
        params.usepytorch = True if "usepytorch" not in params else params.usepytorch
        params.seed = 0 if "seed" not in params else params.seed


        params.batch_size = 64 if "batch_size" not in params else params.batch_size
        params.nhid = 0 if "nhid" not in params else params.nhid
        params.kfold = 5 if "kfold" not in params else params.kfold

        if "classifier" not in params or not params["classifier"]:
            params.classifier = {"nhid": 0}

        assert (
            "nhid" in params.classifier
        ), "Set number of hidden units in classifier config!!"

        self.params = params

        # batcher and prepare
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None

        self.list_tasks = ["BUG"]

    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if isinstance(name, list):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + " not in " + str(self.list_tasks)

        # Original SentEval tasks

        if name == "BUG":
            self.evaluation = BUGEval(
                # tpath + "/downstream/BUG", nclasses=2, seed=self.params.seed
                "","",tpath, nclasses=2, seed=self.params.seed,
            )

        self.params.current_task = name


        self.results = self.evaluation.run(self.params, self.batcher)

        return self.results

    def predict(self,name,modelPath):
        if isinstance(name, list):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + " not in " + str(self.list_tasks)

        # Original SentEval tasks

        if name == "BUG":
            self.evaluation = BUGEval(
                # tpath + "/downstream/BUG", nclasses=2, seed=self.params.seed
                modelPath,"pred",tpath, nclasses=2, seed=self.params.seed,
            )
            self.evaluation.run_pred(self.params, self.batcher)


        self.params.current_task = name


