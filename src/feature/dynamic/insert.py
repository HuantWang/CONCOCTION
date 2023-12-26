import os
import subprocess
import shutil
import sys

#path = os.getcwd()
path = os.path.dirname(os.path.realpath(__file__))
project_path = sys.argv[1]
outlog_path = path +"/instrument/github/out_log"
if os.path.exists(outlog_path):
    shutil.rmtree(outlog_path)
os.mkdir(outlog_path)
str = "python3 "+path+"/instrument/github/main_github.py -path " + project_path + " -logDir " + outlog_path
print(str)
process=subprocess.Popen(str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
exit_code = process.wait()
print(exit_code)
print(stderr.decode())
outlog_err_path = outlog_path + "/insertlog/execInsertPrintCmd"
os.system("cd " + outlog_err_path)
cmd = "cat * |grep \"h\' file not found\""
process1 = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout1, stderr1 = process1.communicate()
print("Which header files were not found:\n", stdout1.decode())
os.system("cd " + path)
