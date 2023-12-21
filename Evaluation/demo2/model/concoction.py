import subprocess,os
import matplotlib.pyplot as plt
import numpy as np
import os

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
   
concoction_detect=r"/homee/concoction/detectionModel"     
class concoction:
    def __init__(self,dataDir,load_saved_model) -> None:
        self.dataDir=dataDir
        self.load_saved_model=load_saved_model
    def run(self):
        dir=concoction_detect
        os.chdir(dir)
        scriptPath="evaluation_bug.py"
        cmd="/root/anaconda3/envs/pytorch1.7.1/bin/python "+scriptPath+" --path_to_data \'"+self.dataDir+"\' --mode test --model_to_load "+self.load_saved_model+" >run.log  2>&1"
        print(cmd)
        execute_shell_script(cmd)
        
        log=concoction_detect+"/result.log"
        draw(log)
        

# if __name__=='__main__':
#     concoction("/home/ExperimentalEvaluation/data/github_0.65","/home/ExperimentalEvaluation/TDSC/11_0.7858546168958742_1434542993.h5").run()
         

        