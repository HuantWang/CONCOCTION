import os
import subprocess
import sys

#file_path = "/home/klee/file/test3/before_insert.txt"
file_path = sys.argv[1]
with open(file_path) as f:
    lines = f.readlines()
    for command in lines:
        if command == "":
            break
        if (command.find("apt") != -1):
            command1 = command.strip() + " -y"
            print(command1.strip())
            process = subprocess.Popen(command1.strip(), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # 监控命令的运行状态和结果
            stdout, stderr = process.communicate()
            exit_code = process.wait()
            if exit_code != 0:
                print("Failed to install dependencies;")
                print("error:" + command.strip())
                print(stderr.decode())
                sys.exit(0)

print("Download dependencies complete!")
# 输出命令的运行结果和状态码
# print("Command:", command.strip())
# print("Exit code:", exit_code)
# print("Stdout:", stdout.decode())
# print("Stderr:", stderr.decode())
