import subprocess,os
def execute_shell_script(cmd):
    
    p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
    stdout_data, stderr_data=p.communicate()
    print(stdout_data.decode('utf-8'))
   
ExperimentalEvaluation=r"/home/ExperimentalEvaluation"     
class LineVul:
    def __init__(self,dataDir,load_saved_model) -> None:
        self.dataDir=dataDir
        self.load_saved_model=load_saved_model
    def run(self):
        dir=os.path.join(ExperimentalEvaluation,'LineVul/linevul')
        os.chdir(dir)
        scriptPath="linevul_main.py"
        cmd="/root/anaconda3/envs/Linevul/bin/python "+scriptPath+" --data_file "+self.dataDir+" --evaluate_during_training --do_train --do_test"
        print(cmd)
        execute_shell_script(cmd)
        

if __name__=='__main__':
    LineVul("/home/ExperimentalEvaluation/data/sard/cwe416","/home/ExperimentalEvaluation/TDSC/11_0.7858546168958742_1434542993.h5").run()
         

        