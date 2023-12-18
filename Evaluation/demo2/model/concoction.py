import subprocess,os
def execute_shell_script(cmd):
    
    p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
    stdout_data, stderr_data=p.communicate()
    print(stdout_data.decode('utf-8'))
   
concoction_detect=r"/home/concoction/detectionModel"     
class concoction:
    def __init__(self,dataDir,load_saved_model) -> None:
        self.dataDir=dataDir
        self.load_saved_model=load_saved_model
    def run(self):
        dir=concoction_detect
        os.chdir(dir)
        scriptPath="evaluation_bug.py"
        cmd="/root/anaconda3/envs/pytorch1.7.1/bin/python "+scriptPath+" --path_to_data \'"+self.dataDir+"\' --mode test --model_to_load "+self.load_saved_model
        print(cmd)
        execute_shell_script(cmd)
        

# if __name__=='__main__':
#     concoction("/home/ExperimentalEvaluation/data/github_0.65","/home/ExperimentalEvaluation/TDSC/11_0.7858546168958742_1434542993.h5").run()
         

        