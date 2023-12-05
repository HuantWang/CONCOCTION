import subprocess,os
def execute_shell_script(cmd):
    
    p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
    stdout_data, stderr_data=p.communicate()
    print(stdout_data.decode('utf-8'))
   
        
class Reveal:
    def __init__(self,dataDir,load_saved_model) -> None:
        self.dataDir=dataDir
        self.load_saved_model=load_saved_model
    def predict(self):
        os.chdir("")
        scriptPath=""
        cmd="python "+scriptPath+" "+self.dataDir+" "+self.load_saved_model
        execute_shell_script(cmd)
        
        # os.chdir("/home/ExperimentalEvaluation/FUNDED_NISL-main/FUNDED/cli")
        # scriptPath="./test.py"
        # cmd="python "+scriptPath+" GGNN GraphBinaryClassification "+self.dataDir+" --storedModel_path "+self.load_saved_model
        # execute_shell_script(cmd)   
# if __name__=="__main__":
#     Funded("/home/ExperimentalEvaluation/FUNDED_NISL-main/FUNDED/data/data/data/cve/badall","/home/ExperimentalEvaluation/FUNDED_NISL-main/FUNDED/cli/trained_model/GGNN_GraphBinaryClassification__2023-02-01_05-36-00_f1=0.800_best.pkl").predict()