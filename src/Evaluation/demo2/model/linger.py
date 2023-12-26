import subprocess,os
import matplotlib.pyplot as plt
import numpy as np
import os

from .util import util
ROOT_PATH=util.getRootpath()
ExperimentalEvaluation=ROOT_PATH+"/Evaluation/ExperimentalEvaluation"  

def draw(log):
    results={}
    with open(log,"r") as f:
        lines=f.readlines()
        for line in lines:
            key, value = line.split(',')
            value=value.replace("\n","")                  
            results[key]=float(value)
            
    keys = np.array(list(results.keys()))
    values = np.array(list(results.values()))
    print(keys)
    print(values)
    
    bar_width = 0.4  
    colors=['#4D1E17','#B43A24','#D89F8A','#F6E1C6']
    plt.bar(keys,values,width=bar_width,color=colors)
    plt.show()
def execute_shell_script(cmd):
    
    p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
    stdout_data, stderr_data=p.communicate()
    print(stdout_data.decode('utf-8'))
   

class Liger:
    def __init__(self,dataDir,load_saved_model) -> None:
        self.dataDir=dataDir
        self.load_saved_model=load_saved_model
    def run(self):
        dir=os.path.join(ExperimentalEvaluation,'linger_finall')
        os.chdir(dir)
        scriptPath="run_group.py"
        cmd="/root/anaconda3/envs/liger/bin/python "+scriptPath+" --data_file \'"+self.dataDir+"\' --model_to_load "+self.load_saved_model+" >run.log  2>&1"
        print(cmd)
        execute_shell_script(cmd)
        
        log=ExperimentalEvaluation+"/linger_finall/result.log"
        draw(log)
        #python /home/ExperimentalEvaluation/lingerwj/run_group.py  --train_data_file /home/ExperimentalEvaluation/data/github_0.6_new/test --model_to_load /home/ExperimentalEvaluation/lingerwj/savedmodel/0.7207637231503581.cpkt

# if __name__=='__main__':
#     LineVul("/home/ExperimentalEvaluation/data/sard/cwe416","/home/ExperimentalEvaluation/TDSC/11_0.7858546168958742_1434542993.h5").run()
         

        