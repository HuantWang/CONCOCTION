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
   
   
class vuldeepecker:
    def __init__(self,dataDir,load_saved_model) -> None:
        self.dataDir=dataDir
        self.load_saved_model=load_saved_model
    def run(self):
        dir=os.path.join(ExperimentalEvaluation,'vuldeepker')
        os.chdir(dir)
        scriptPath="vuldeepecker.py"
        cmd="/root/anaconda3/envs/vuldeepecker1/bin/python "+scriptPath+" --data_path \'"+self.dataDir+"\' --mode pre --model_to_load "+self.load_saved_model+" >run.log  2>&1"
        execute_shell_script(cmd)
        
        log=ExperimentalEvaluation+"/vuldeepker/result.log"
        draw(log)
        
        

# if __name__=='__main__':
#     vuldeepecker("/home/ExperimentalEvaluation/vuldeepker/datas","/home/ExperimentalEvaluation/TDSC/11_0.7858546168958742_1434542993.h5").run()
         

        