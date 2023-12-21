import subprocess,os,time

ROOT_PATH=r"/homee" 
python="/root/anaconda3/envs/pytorch1.7.1/bin/python"    
def execute_shell_script(cmd):
    p = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
    stdout_data, stderr_data=p.communicate()
    print(stdout_data.decode('utf-8'))
    return p
    
class concoction():
    def __init__(self) -> None:
        pass
    
    def trainDetectModel():
        current=ROOT_PATH
        path=os.path.join(current,'Evaluation/ExperimentalEvaluation/data/github_0.6_new/train')
        dir=os.path.join(current,'concoction/detectionModel')
        cmd=f"cd {dir} && {python} evaluation_bug.py --path_to_data {path} --mode train"
        print("Training the detection Model...")
        execute_shell_script(cmd)
        
    def staticRepre(data_path):
        dir=ROOT_PATH+"/concoction/pretrainedModel/staticRepresentation"
        outputPath=ROOT_PATH+"/concoction/data/output_static.txt"
        storedPath=ROOT_PATH+"/concoction/pretrainedModel/staticRepresentation/trainedModel"
        if os.path.exists(outputPath):
            os.remove(outputPath)
        if not os.path.exists(os.path.dirname(outputPath)):
            os.mkdir(os.path.dirname(outputPath))
        cmd1=f"cd {dir} && {python} preprocess.py --data_path {data_path} --output_path {outputPath}"
        cmd2=f"cd {dir} && {python} train.py --model_name_or_path graphcodebert-base --train_data_file {outputPath} --per_device_train_batch_size 4 --do_train --output_dir {storedPath} --mlm --overwrite_output_dir --line_by_line"
        print(" Pretraining the Representation Models......")
        p1=execute_shell_script(cmd1)
        p1.wait()
        execute_shell_script(cmd2)
        print(f" Saving model checkpoint to {storedPath}")

        pass
    
    def dynamicRepre(data_path):
        #  Pretrain Representation Models (Here we use Simcse like sec 3.4)
        #preprocess
        current=ROOT_PATH
        outputPath=os.path.join(current,'concoction/data/output_dynamic.txt')
        dir=os.path.join(current,'concoction/pretrainedModel/dynamicRepresentation')
        storedPath=ROOT_PATH+"/concoction/pretrainedModel/dynamicRepresentation/trainedModel"
        
        if os.path.exists(outputPath):
            os.remove(outputPath)
        if not os.path.exists(os.path.dirname(outputPath)):
            os.mkdir(os.path.dirname(outputPath))
        
        cmd1=f"cd {dir} && {python} preprocess.py --data_path {data_path} --output_path {outputPath} "
        cmd2=f"cd {dir} && {python} train.py --model_name_or_path bert-base-uncased     --train_file {outputPath}   --output_dir {storedPath}    --num_train_epochs 1     --per_device_train_batch_size 32     --learning_rate 3e-5     --max_seq_length 32      --metric_for_best_model stsb_spearman  --load_best_model_at_end     --eval_steps 2     --pooler_type cls     --mlp_only_train     --overwrite_output_dir     --temp 0.05     --do_train"
        cmd2=cmd2+" > dynamicRepre.log 2>&1"
        print(" Preprocesing the data......")
        p1=execute_shell_script(cmd1)
        p1.wait()
        time.sleep(2)
        print(" Pretraining the Representation Models......")
        execute_shell_script(cmd2)
        print(f" Saving model checkpoint to {storedPath}")
        
    def showDynamicRepre():
        print(" Show the Trained Model(Dynamic code Infomation Representation Model)")
        dir=os.path.join(ROOT_PATH,'/homee/Evaluation/demo1')
        cmd=f"cd {dir} && {python} showDynaRepr.py"
        execute_shell_script(cmd)
       
    def showStaticRepre():
        print(" Show the Trained Model(Static code Infomation Representation Model)")
        dir=os.path.join(ROOT_PATH,'/homee/Evaluation/demo1')
        cmd=f"cd {dir} && {python} showStaticRepr.py"
        execute_shell_script(cmd)
        

    def showDetectModel():
        print("Show the Trained Model (Load trained model and test on test case)")
        current=ROOT_PATH
        path=os.path.join(current,'Evaluation/ExperimentalEvaluation/data/github_0.6_new/test')
        dir=os.path.join(current,'concoction/detectionModel')
        model_to_load=os.path.join(current,'Evaluation/ExperimentalEvaluation/Concoction/saved_models/github.h5')
        cmd=f"cd {dir} && {python} evaluation_bug.py  --model_to_load {model_to_load} --path_to_data {path}  --mode test"
        execute_shell_script(cmd)
        
    def extractPathRepre():
        print("Extracting Execution path representation...")
        current=ROOT_PATH
        path=os.path.join(current,'concoction/data/dataset0')
        storedDir=os.path.join(current,'concoction/data/feature_path')
        dir=os.path.join(current,'concoction/pathSelection')

           
        if not os.path.exists(storedDir):
            os.mkdir(storedDir)
        cmd=f"cd {dir} && {python} preprocess.py --data_path {path} --stored_path {storedDir}"
        execute_shell_script(cmd)
        
        
    def pathSelect():
       
        print("Active learning for path selection...")
        current=ROOT_PATH
        storedDir=os.path.join(current,'concoction/data/feature_path')
        final_storedDir=os.path.join(current,'concoction/data/feature_path_text')
        dir=os.path.join(current,'concoction/pathSelection')


        if not os.path.exists(storedDir):
            os.mkdir(storedDir)
        cmd=f"cd {dir} && {python} train.py --data_path {storedDir} --stored_path {final_storedDir}"
        execute_shell_script(cmd)
        cmd2=f"bash {current}/Evaluation/demo1/showSelectedPath.sh"
        execute_shell_script(cmd2)

    def symbolicOnPath():
        # Symbolic execution for chosen paths (Sec. 3.6.3)
        current=ROOT_PATH
        print("Symbolic execution for chosen paths...")
        path=os.path.join(current,'feature/dynamic/map.py')
        dataset=os.path.join(current,'concoction/data/dataset')
        feature_path_text=os.path.join(current,'concoction/data/feature_path_text')
        final_path=os.path.join(current,'concoction/data/feature_result')
        cmd=f"{python} {path} {feature_path_text} {dataset} {final_path}   "
        execute_shell_script(cmd)
        
    def showExampDynaInfo():
        dir=''
        cmd="sh /homee/Evaluation/demo1/showFile.sh dir"