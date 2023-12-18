import subprocess,os
def execute_shell_script(cmd):
    
    p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
    stdout_data, stderr_data=p.communicate()
    print(stdout_data.decode('utf-8'))
   
ExperimentalEvaluation=r"/homee/Evaluation/ExperimentalEvaluation"     
class Liger:
    def __init__(self,dataDir,load_saved_model) -> None:
        self.dataDir=dataDir
        self.load_saved_model=load_saved_model
    def run(self):
        dir=os.path.join(ExperimentalEvaluation,'linger_finall')
        os.chdir(dir)
        scriptPath="run_group.py"
        cmd="/root/anaconda3/envs/liger/bin/python "+scriptPath+" --data_file \'"+self.dataDir+"\' --model_to_load "+self.load_saved_model
        print(cmd)
        execute_shell_script(cmd)
        #python /home/ExperimentalEvaluation/lingerwj/run_group.py  --train_data_file /home/ExperimentalEvaluation/data/github_0.6_new/test --model_to_load /home/ExperimentalEvaluation/lingerwj/savedmodel/0.7207637231503581.cpkt

# if __name__=='__main__':
#     LineVul("/home/ExperimentalEvaluation/data/sard/cwe416","/home/ExperimentalEvaluation/TDSC/11_0.7858546168958742_1434542993.h5").run()
         

        