import subprocess,os
def execute_shell_script(cmd):
    
    p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
    stdout_data, stderr_data=p.communicate()
    print(stdout_data.decode('utf-8'))
   
ExperimentalEvaluation=r"/homee/Evaluation/ExperimentalEvaluation"     
class vuldeepecker:
    def __init__(self,dataDir,load_saved_model) -> None:
        self.dataDir=dataDir
        self.load_saved_model=load_saved_model
    def run(self):
        dir=os.path.join(ExperimentalEvaluation,'vuldeepker')
        os.chdir(dir)
        scriptPath="vuldeepecker.py"
        cmd="/root/anaconda3/envs/vuldeepecker1/bin/python "+scriptPath+" --data_path \'"+self.dataDir+"\' --mode pre --model_to_load "+self.load_saved_model+" 2>run.log"
        print(cmd)
        execute_shell_script(cmd)
        

# if __name__=='__main__':
#     vuldeepecker("/home/ExperimentalEvaluation/vuldeepker/datas","/home/ExperimentalEvaluation/TDSC/11_0.7858546168958742_1434542993.h5").run()
         

        