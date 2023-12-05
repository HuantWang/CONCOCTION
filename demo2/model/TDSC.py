import subprocess,os
def execute_shell_script(cmd):
    
    p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
    stdout_data, stderr_data=p.communicate()
    print(stdout_data.decode('utf-8'))
   
ExperimentalEvaluation=r"/home/ExperimentalEvaluation"     
class TDSC:
    def __init__(self,dataDir,load_saved_model) -> None:
        self.dataDir=dataDir
        self.load_saved_model=load_saved_model
    def run(self):
        dir=os.path.join(ExperimentalEvaluation,'TDSC')
        os.chdir(dir)
        scriptPath="TDSC.py"
        cmd="/root/anaconda3/envs/vuldeepecker1/bin/python "+scriptPath+" --data_path "+self.dataDir+" --mode train"
        print(cmd)
        execute_shell_script(cmd)
        
# if __name__=="__main__":
#     Funded("/home/ExperimentalEvaluation/FUNDED_NISL-main/FUNDED/data/data/data/cve/badall","/home/ExperimentalEvaluation/FUNDED_NISL-main/FUNDED/cli/trained_model/GGNN_GraphBinaryClassification__2023-02-01_05-36-00_f1=0.800_best.pkl").predict()

if __name__=='__main__':
    TDSC("/home/ExperimentalEvaluation/TDSC/datas","/home/ExperimentalEvaluation/TDSC/11_0.7858546168958742_1434542993.h5").predict()
         

        