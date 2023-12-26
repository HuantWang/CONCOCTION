import sys,os
current_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_folder)
# from model.Funded import Funded
# from model.devign import Devign
# from model.reveal import Reveal
# from model.codebert import CodeBERT
# from model.graphcodebert import GraphCodeBERT
# from model.regvd import ReGVD
from model.TDSC import TDSC
from model.vuldeepecker import vuldeepecker
from model.concoction import concoction
from model.LineVul import LineVul
# from model.ContraFlow import ContraFlow
from model.linger import Liger

import sys,os
script_dir = os.path.dirname(os.path.abspath(__file__))
top_level_path=script_dir
while True:
    top_level_path, tail = os.path.split(top_level_path)
    if tail == "Evaluation":
        top_level_path=os.path.join(top_level_path,tail)
        break
sys.path.append(top_level_path)
print(top_level_path)
from demo1.setRootpath import getRootpath
ROOT_PATH=getRootpath()
ExperimentalEvaluation=ROOT_PATH+"/Evaluation/ExperimentalEvaluation"  

class EvaluationSard:
    def __init__(self,dataset,method) -> None:
        self.cwetype=""
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
        load_saved_model=os.path.join(ExperimentalEvaluation,'new_Reveal/DynamicBugDetection-Reveal/modles/test-checkpoint/78.87323943661973.pkl')
        print(f"method:vuldeepecker\nload_saved_model:{load_saved_model}\ndataDir:{self.dataDir}")
        Reveal(self.dataDir,load_saved_model).predict()

    def codebert_run(self):
        load_saved_model=os.path.join(ExperimentalEvaluation,'CodeBert/Defect-detection/code/pre-saved_models/test-checkpoint/0.9730639730639731_123456.bin')
        print(f"method:vuldeepecker\nload_saved_model:{load_saved_model}\ndataDir:{self.dataDir}")
        CodeBERT(self.dataDir,load_saved_model).predict()
    
    def graphcodebert_run(self):
        load_saved_model = r""
        GraphCodeBERT(self.dataDir,load_saved_model).predict()
    
    def regvd_run(self):
        load_saved_model = r""
        ReGVD(self.dataDir,load_saved_model).predict()
        
    def TDSC_run(self):
        load_saved_model=os.path.join(ExperimentalEvaluation,"TDSC/saved_models/"+self.cwetype+".h5")
        print(f"method:tdsc\nload_saved_model:{load_saved_model}\ndataDir:{self.dataDir}")
        TDSC(self.dataDir,load_saved_model).run()
    def vuldeepecker_run(self):
        load_saved_model=os.path.join(ExperimentalEvaluation,"vuldeepker/saved_models/"+self.cwetype+".h5")
        print(f"method:vuldeepecker\nload_saved_model:{load_saved_model}\ndataDir:{self.dataDir}")
        vuldeepecker(self.dataDir,load_saved_model).run()
    def concoction_run(self):
        load_saved_model=os.path.join(ExperimentalEvaluation,"Concoction/saved_models/"+self.cwetype+".h5")
        print(f"method:concoction\nload_saved_model:{load_saved_model}\ndataDir:{self.dataDir}")
        concoction(self.dataDir,load_saved_model).run()
    def LineVul_run(self):
        load_saved_model=os.path.join(ExperimentalEvaluation,"lineVul/linevul/saved_modelss/"+self.cwetype+".h5")
        print(f"method:linvul\nload_saved_model:{load_saved_model}\ndataDir:{self.dataDir}")
        LineVul(self.dataDir,load_saved_model).run() 
    def Liger_run(self):
        load_saved_model=os.path.join(ExperimentalEvaluation,"linger_finall/saved_models/"+self.cwetype+".cpkt")
        print(f"method:liger\nload_saved_model:{load_saved_model}\ndataDir:{self.dataDir}")
        Liger(self.dataDir,load_saved_model).run()
        
    def ContraFlow_run(self):
        load_saved_model=''
        ContraFlow(self.dataDir,load_saved_model).run()
        
    def run(self):
        sardDataset={
            "CWE-416":os.path.join(ExperimentalEvaluation,"data/sard/cwe416_test"),
            "CWE-789":os.path.join(ExperimentalEvaluation,"data/sard/cwe789_test"),
            "CWE-78":os.path.join(ExperimentalEvaluation,"data/sard/cwe78_124_test"),
            "CWE-124":os.path.join(ExperimentalEvaluation,"data/sard/cwe78_124_test"),
            "CWE-122":os.path.join(ExperimentalEvaluation,"data/sard/cwe122(1164-1164)/test"),
            "CWE-190":os.path.join(ExperimentalEvaluation,"data/sard/cwe190/test"),
            "CWE-191":os.path.join(ExperimentalEvaluation,"data/sard/cwe191(1101-1101)/test"),
            "CWE-126":os.path.join(ExperimentalEvaluation,"data/sard/cwe126(274-274)/test")
        }
        sardType={
            "CWE-416":"cwe416",
            "CWE-789":"cwe789",
            "CWE-78":"cwe78-124",
            "CWE-124":"cwe78-124",
            "CWE-122":"cwe122",
            "CWE-190":"cwe190",
            "CWE-191":"cwe191",
            "CWE-126":"cwe126"
        }

        method=["funded","devign","reveal","codebert","graphcodebert","regvd","TDSC","vuldeepecker","concoction","LineVul","ContraFlow","Liger",""]
        if self.dataset in sardDataset.keys():
            self.dataDir=sardDataset[self.dataset]
            self.cwetype=sardType[self.dataset]
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
            if self.method=="concoction":
                self.concoction_run()
            if self.method=="LineVul":
                self.LineVul_run()
            if self.method=="Liger":
                self.Liger_run()
            if self.method=="ContraFlow":
                self.ContraFlow_run()
            
        else:
            print('unknown method.')
            return


class EvaluationGithub:
    def __init__(self,dataset,method) -> None:
        self.dataset=dataset
        self.dataDir=""
        self.method=method
    # def funded_run(self):
    #     load_saved_model=r"../ExperimentalEvaluation/FUNDED_NISL-main/FUNDED/cli/trained_model/GGNN_GraphBinaryClassification__2023-02-01_05-36-00_f1=0.800_best.pkl"
    #     Funded(self.dataDir,load_saved_model).predict()
        
    # def devign_run(self):
    #     load_saved_model=r"../ExperimentalEvaluation/Devign/models/ours/GGNNSumModel86.02150537634408_42.bin"
    #     Devign(self.dataDir,load_saved_model).predict()
        
    def reveal_run(self):
        load_saved_model=os.path.join(ExperimentalEvaluation,'new_Reveal/DynamicBugDetection-Reveal/modles/test-checkpoint/78.87323943661973.pkl')
        # load_saved_model=r"../ExperimentalEvaluation/new_Reveal/DynamicBugDetection-Reveal/modles/test-checkpoint/78.87323943661973.pkl"
        print(f"method:reveal\nload_saved_model:{load_saved_model}\ndataDir:{self.dataDir}")
        Reveal(self.dataDir,load_saved_model).predict()

    def codebert_run(self):
        load_saved_model=os.path.join(ExperimentalEvaluation,'CodeBert/Defect-detection/code/pre-saved_models/test-checkpoint/0.9730639730639731_123456.bin')
        print(f"method:codebert\nload_saved_model:{load_saved_model}\ndataDir:{self.dataDir}")
        CodeBERT(self.dataDir,load_saved_model).predict()

    def TDSC_run(self):
        load_saved_model=os.path.join(ExperimentalEvaluation,"TDSC/saved_models/github.h5")
        print(f"method:tdsc\nload_saved_model:{load_saved_model}\ndataDir:{self.dataDir}")
        TDSC(self.dataDir,load_saved_model).run()
    def vuldeepecker_run(self):
        load_saved_model=os.path.join(ExperimentalEvaluation,'vuldeepker/saved_models/github.h5')
        print(f"method:vuldeepecker\nload_saved_model:{load_saved_model}\ndataDir:{self.dataDir}")
        vuldeepecker(self.dataDir,load_saved_model).run()
    def concoction_run(self):
        load_saved_model=os.path.join(ExperimentalEvaluation,'Concoction/saved_models/github.h5')
        print(f"method:concoction\nload_saved_model:{load_saved_model}\ndataDir:{self.dataDir}")
        concoction(self.dataDir,load_saved_model).run()
    def LineVul_run(self):
        load_saved_model=os.path.join(ExperimentalEvaluation,'lineVul/linevul/saved_modelss/0.727699530516432_123456_1.h5')
        print(f"method:linevul\nload_saved_model:{load_saved_model}\ndataDir:{self.dataDir}")
        LineVul(self.dataDir,load_saved_model).run()
    def Liger_run(self):
        load_saved_model=os.path.join(ExperimentalEvaluation,'linger_finall/saved_models/0.8540540540540541.cpkt')
        print(f"method:liger\nload_saved_model:{load_saved_model}\ndataDir:{self.dataDir}")
        Liger(self.dataDir,load_saved_model).run()
    def ContraFlow_run(self):
        load_saved_model=''
        ContraFlow(self.dataDir,load_saved_model).run()
    
        
    def run(self):
        path=os.path.join(ExperimentalEvaluation,"data/github_0.6_new/test")
        GithubDataset={#todo github dataset
            "Github":path
        }
        method=["funded","devign","reveal","TDSC","vuldeepecker","concoction","LineVul","ContraFlow","Liger","codebert"]
        # method=["reveal"]
        if self.dataset == 'Github':# todo: Github datadir
            self.dataDir=GithubDataset["Github"]
        else:
            print(f"unknown dataset: {self.dataset}")
            return
            
        if self.method in method:
            # if self.method=="funded":
            #     self.funded_run()
            # if self.method=="devign":
            #     self.devign_run()
            if self.method=="reveal":
                self.reveal_run()
            if self.method=="codebert":
                self.codebert_run()
            if self.method=="TDSC":
                self.TDSC_run()
            if self.method=="vuldeepecker":
                self.vuldeepecker_run()
            if self.method=="concoction":
                self.concoction_run()
            if self.method=="LineVul":
                self.LineVul_run()
            # if self.method=="Liger":
            #     self.Liger_run()
            # if self.method=="ContraFlow":
            #     self.ContraFlow_run()
            
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
    # EvaluationSard("CWE-126", "TDSC").run()
    # print("---------------------------------------------")
    # EvaluationGithub("Github","vuldeepecker").run()
    # EvaluationSard("CWE-126", "vuldeepecker").run()
    # print("---------------------------------------------")
    # EvaluationGithub("Github","concoction").run()
    # EvaluationSard("CWE-78", "concoction").run()
    
    # EvaluationGithub("Github","ContraFlow").run()
    # print("---------------------------------------------")
    # EvaluationGithub("Github","LineVul").run()
    # print("---------------------------------------------")
    # EvaluationSard("CWE-78", "LineVul").run()
    # print("---------------------------------------------")
    # # EvaluationSard("CWE-416", "vuldeepecker").run()
    # print("---------------------------------------------")
    # EvaluationGithub("Github","Liger").run()
    # print("---------------------------------------------")
    # EvaluationSard("CWE-191", "Liger").run()
    print("---------------------------------------------")
    EvaluationGithub("Github","codebert").run()


    
