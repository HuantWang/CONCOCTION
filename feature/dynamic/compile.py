import os
import subprocess
import shutil
import sys

file_path = sys.argv[1]
txt_path= sys.argv[2]
os.system("chmod 777 -R " + file_path)
os.chdir(file_path)
print(os.getcwd())
os.environ["LLVM_COMPILER"] = "clang"
#os.system("export WLLVM_OUPUT=DEBUG")
os.environ["WLLVM_OUPUTR"] = "DEBUG"
#print(os.environ["WLLVM_OUPUTR"])
#os.system("export CC=/usr/local/bin/wllvm")
os.environ["CC"] = "/usr/local/bin/wllvm"
#os.system("export CXX=/usr/local/bin/wllvm++")
os.environ["CXX"] = "/usr/local/bin/wllvm++"

#os.system("export CFLAGS=\"-g -O1 -Xclang -disable-llvm-passes -DNO_STRING_INLINES -D_FORTIFY_SOURCE=0 -UOPTIMIZE__\"")
os.environ["CFLAGS"] = "-g -O1 -Xclang -disable-llvm-passes -DNO_STRING_INLINES -D_FORTIFY_SOURCE=0 -UOPTIMIZE__"
with open(txt_path) as f:
    lines = f.readlines()
    for command in lines:
        # 使用os.system()函数执行命令
        process = subprocess.Popen(command.strip(), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 监控命令的运行状态和结果
        stdout, stderr = process.communicate()
        exit_code = process.wait()
        if exit_code != 0:
            print("Failed to compile;")
            print("error:" + command.strip())
            print(stderr.decode())
            sys.exit(0)
print("Compile complete!")