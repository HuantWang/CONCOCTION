import subprocess,os
def execute_shell_script(cmd):
    
    p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
    stdout_data, stderr_data=p.communicate()
    print(stdout_data.decode('utf-8'))
   
        
class ReGVD:
    def __init__(self,dataDir,load_saved_model) -> None:
        self.dataDir=dataDir
        self.load_saved_model=load_saved_model
    def predict(self):
        os.chdir("/home/ExperimentalEvaluation/ReGVD/data/")
        os.system("/root/anaconda3/envs/vuldeepecker/bin/python /home/ExperimentalEvaluation/ReGVD/data/preprocess.py " + self.dataDir)
        os.chdir("/home/ExperimentalEvaluation/ReGVD/code")
        scriptPath="./run_new.py"
        cmd="/root/anaconda3/envs/vuldeepecker/bin/python "+scriptPath + " --output_dir=./saved_models --model_type=roberta --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --do_train --do_eval --do_test "+"--train_data_file=../data/train.jsonl --eval_data_file=../data/valid.jsonl --test_data_file=../data/test.jsonl --epoch 5 --block_size 128 --train_batch_size 32 --eval_batch_size 32 --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456"
        execute_shell_script(cmd)
        
# if __name__=="__main__":
#     Funded("/home/ExperimentalEvaluation/FUNDED_NISL-main/FUNDED/data/data/data/cve/badall","/home/ExperimentalEvaluation/FUNDED_NISL-main/FUNDED/cli/trained_model/GGNN_GraphBinaryClassification__2023-02-01_05-36-00_f1=0.800_best.pkl").predict()
        

        