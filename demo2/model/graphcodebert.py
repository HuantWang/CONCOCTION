import subprocess,os
def execute_shell_script(cmd):
    
    p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
    stdout_data, stderr_data=p.communicate()
    print(stdout_data.decode('utf-8'))
   
        
class GraphCodeBERT:
    def __init__(self,dataDir,load_saved_model) -> None:
        self.dataDir=dataDir
        self.load_saved_model=load_saved_model
    def predict(self):
        os.chdir("/home/ExperimentalEvaluation/GraphCodeBert/data")
        os.system("/root/anaconda3/envs/vuldeepecker/bin/python /home/ExperimentalEvaluation/GraphCodeBert/data/preprocess.py " + self.dataDir)
        os.chdir("/home/ExperimentalEvaluation/GraphCodeBert/")
        scriptPath="./run_bug_acc.py"
        cmd="/root/anaconda3/envs/vuldeepecker/bin/python "+scriptPath + " --output_dir=./saved_models --config_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base --tokenizer_name=microsoft/graphcodebert-base --do_train --do_eval --do_test "+"--train_data_file=data/train.jsonl --eval_data_file=data/valid.jsonl --test_data_file=data/test.jsonl --epoch 5 --code_length 128 --data_flow_length 128 --train_batch_size 32 --eval_batch_size 32 --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456"
        execute_shell_script(cmd)
        
# if __name__=="__main__":
#     Funded("/home/ExperimentalEvaluation/FUNDED_NISL-main/FUNDED/data/data/data/cve/badall","/home/ExperimentalEvaluation/FUNDED_NISL-main/FUNDED/cli/trained_model/GGNN_GraphBinaryClassification__2023-02-01_05-36-00_f1=0.800_best.pkl").predict()
        

        