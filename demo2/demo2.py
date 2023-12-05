import sys,os
current_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_folder)
from model.Funded import Funded
from model.devign import Devign
from model.reveal import Reveal
from model.codebert import CodeBERT
from model.graphcodebert import GraphCodeBERT
from model.regvd import ReGVD
from model.TDSC import TDSC
from model.vuldeepecker import vuldeepecker
from model.concoction import concoction
from model.LineVul import LineVul
from model.ContraFlow import ContraFlow
import os
ExperimentalEvaluation=r"/home/ExperimentalEvaluation"     
class EvaluationSard:
    def __init__(self,dataset,method) -> None:
        self.dataset=dataset
        self.dataDir=""
        self.method=method
    def funded_run(self):
        load_saved_model=r"../ExperimentalEvaluation/FUNDED_NISL-main/FUNDED/cli/trained_model/GGNN_GraphBinaryClassification__2023-02-01_05-36-00_f1=0.800_best.pkl"
        Funded(self.dataDir,load_saved_model).predict()
        
    def devign_run(self):
        load_saved_model=r"../ExperimentalEvaluation/Devign/models/ours/GGNNSumModel86.02150537634408_42.bin"
        Devign(self.dataDir,load_saved_model,self.dataset).predict()
        
    def reveal_run(self):
        load_saved_model=r"../ExperimentalEvaluation/pkl"
        Reveal(self.dataDir,load_saved_model).predict()

    def codebert_run(self):
        load_saved_model = r""
        CodeBERT(self.dataDir,load_saved_model).predict()
    
    def graphcodebert_run(self):
        load_saved_model = r""
        GraphCodeBERT(self.dataDir,load_saved_model).predict()
    
    def regvd_run(self):
        load_saved_model = r""
        ReGVD(self.dataDir,load_saved_model).predict()
        
    def TDSC_run(self):
        load_saved_model=''
        TDSC(self.dataDir,load_saved_model).run()
    def vuldeepecker_run(self):
        load_saved_model=''
        vuldeepecker(self.dataDir,load_saved_model).run()
    def concoction_run(self):
        load_saved_model=''
        concoction(self.dataDir,load_saved_model).run()
    def LineVul_run(self):
        load_saved_model=''
        LineVul(self.dataDir,load_saved_model).run()
    def ContraFlow_run(self):
        load_saved_model=''
        ContraFlow(self.dataDir,load_saved_model).run()
        
    def run(self):
        sardDataset={
            "CWE-416":os.path.join(ExperimentalEvaluation,"data/sard/cwe416"),
            "CWE-789":os.path.join(ExperimentalEvaluation,"data/sard/cwe789"),
            "CWE-78":os.path.join(ExperimentalEvaluation,"data/sard/cwe78"),
        }

        method=["funded","devign","reveal","codebert","graphcodebert","regvd","TDSC","vuldeepecker","concoction","LineVul","ContraFlow"]
        if self.dataset in sardDataset.keys():
            self.dataDir=sardDataset[self.dataset]
        else:
            print(f"unknown dataset: {self.dataset}")
            return
            
        if self.method in method:
            if self.method=="funded":
                self.funded_run()
            if self.method=="devign":
                #self.dataDir = self.dataset
                self.devign_run()
            if self.method=="reveal":
                self.reveal_run()
            if self.method == "codebert":
                self.codebert_run()
            if self.method == "graphcodebert":
                self.graphcodebert_run()
            if self.method == "regvd":
                self.regvd_run()
            if self.method=="TDSC":
                self.TDSC_run()
            if self.method=="vuldeepecker":
                self.vuldeepecker_run()
            
        else:
            print('unknown method.')
            return


class EvaluationGithub:
    def __init__(self,dataset,method) -> None:
        self.dataset=dataset
        self.dataDir=""
        self.method=method
    def funded_run(self):
        load_saved_model=r"../ExperimentalEvaluation/FUNDED_NISL-main/FUNDED/cli/trained_model/GGNN_GraphBinaryClassification__2023-02-01_05-36-00_f1=0.800_best.pkl"
        Funded(self.dataDir,load_saved_model).predict()
        
    def devign_run(self):
        load_saved_model=r"../ExperimentalEvaluation/Devign/models/ours/GGNNSumModel86.02150537634408_42.bin"
        Devign(self.dataDir,load_saved_model).predict()
        
    def reveal_run(self):
        load_saved_model=r"../ExperimentalEvaluation/pkl"
        Reveal(self.dataDir,load_saved_model).predict()
    def TDSC_run(self):
        load_saved_model=os.path.join(ExperimentalEvaluation,"TDSC/11_0.7858546168958742_1434542993.h5")
        TDSC(self.dataDir,load_saved_model).run()
    def vuldeepecker_run(self):
        load_saved_model=''
        vuldeepecker(self.dataDir,load_saved_model).run()
    def concoction_run(self):
        load_saved_model=''
        concoction(self.dataDir,load_saved_model).run()
    def LineVul_run(self):
        load_saved_model=''
        LineVul(self.dataDir,load_saved_model).run()
    def ContraFlow_run(self):
        load_saved_model=''
        ContraFlow(self.dataDir,load_saved_model).run()
    
        
    def run(self):
        path=os.path.join(ExperimentalEvaluation,"data/github_0.65")
        GithubDataset={#todo github dataset
            "Github":path
        }
        method=["funded","devign","reveal","TDSC","vuldeepecker","concoction","LineVul","ContraFlow"]

        if self.dataset == 'Github':# todo: Github datadir
            self.dataDir=GithubDataset["Github"]
        else:
            print(f"unknown dataset: {self.dataset}")
            return
            
        if self.method in method:
            if self.method=="funded":
                self.funded_run()
            if self.method=="devign":
                self.devign_run()
            if self.method=="reveal":
                self.reveal_run()
            if self.method=="TDSC":
                self.TDSC_run()
            if self.method=="vuldeepecker":
                self.vuldeepecker_run()
            if self.method=="concoction":
                self.concoction_run()
            if self.method=="LineVul":
                self.LineVul_run()
            if self.method=="ContraFlow":
                self.ContraFlow_run()
            
        else:
            print('unknown method.')
            return

if __name__=="__main__":
    #EvaluationSard("CWE-416","funded").run()
    #EvaluationSard("CWE-416","devign").run()
    #EvaluationSard("CWE-416","reveal").run()
    #EvaluationSard("CWE-416", "codebert").run()
    #EvaluationSard("CWE-416", "graphcodebert").run()
    # EvaluationSard("CWE-789", "regvd").run()
    # print("---------------------------------------------")
    # EvaluationGithub("Github","TDSC").run()
    print("---------------------------------------------")
    EvaluationGithub("Github","vuldeepecker").run()
    # print("---------------------------------------------")
    # # EvaluationSard("CWE-416", "TDSC").run()
    # EvaluationGithub("Github","ContraFlow").run()
    # print("---------------------------------------------")
    # EvaluationGithub("Github","LineVul").run()
    # print("---------------------------------------------")
    # # EvaluationSard("CWE-416", "vuldeepecker").run()
    # EvaluationGithub("Github","concoction").run()


    
