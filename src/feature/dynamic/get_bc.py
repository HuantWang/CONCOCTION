import os
import subprocess
import shutil
import sys

project_path = sys.argv[1]
do_path = sys.argv[2]
# 换根目录，便于移植
# path=os.getcwd()
path = os.path.dirname(os.path.realpath(__file__))
replay_path = path + "/dynamic"

if os.path.exists(replay_path):
    shutil.rmtree(replay_path)
os.mkdir(replay_path)

with open(do_path) as f:
    lines = f.readlines()
    if len(lines) % 2 != 0:
        print("Command error1")
        exit(0)
    for i, command in enumerate(lines):
        if i % 2 == 0:
            # 获取bc文件
            flag=0
            exe_path = project_path + "/" + command
            os.chdir(os.path.dirname(exe_path))
            os.environ["LLVM_COMPILER"] = "clang"
            os.environ["WLLVM_OUPUTR"] = "DEBUG"
            os.environ["CC"] = "/usr/local/bin/wllvm"
            os.environ["CXX"] = "/usr/local/bin/wllvm++"
            os.environ[
                "CFLAGS"] = "-g -O1 -Xclang -disable-llvm-passes -DNO_STRING_INLINES -D_FORTIFY_SOURCE=0 -UOPTIMIZE__"
            str2 = "extract-bc " + exe_path.strip("\n")
            process = subprocess.Popen(str2.strip(), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # # 监控命令的运行状态和结果
            stdout, stderr = process.communicate()
            exit_code = process.wait()
            if exit_code != 0:
                print("Cann't get bc file;")
                print("error:" + command.strip())
                print(stderr.decode())
                sys.exit(0)
            print("Get bc!")

            #检测有没有第三方库
            executable_path = exe_path.strip("\n")
            output = subprocess.check_output(["ldd", executable_path]).decode()
            lines = output.strip().split("\n")
            for line in lines:
                parts = line.split()
                # print(parts)
                if len(parts) > 2 and "=>" in line and project_path in line:
                    #动态库在项目文件夹
                    #print("1")
                    executable_path2=parts[2]
                    output2 = subprocess.check_output(["file", executable_path2]).decode()
                    file_d_path=""
                    if "symbolic link" in output2:
                        #ldd指向链接
                        #print("2")
                        parts2 = output2.split()
                        file_d=parts2[-1]
                        d_paths=parts[2].split("/")
                        for i,d_path in enumerate(d_paths):
                            file_d_path=file_d_path+d_path+"/"
                            if i == (len(d_paths) - 2):
                                break
                        file_d_path=file_d_path+file_d
                    else:
                        #ldd指向动态库
                        #print("3")
                        file_d_path=executable_path2
                    str3 = "extract-bc " + file_d_path.strip("\n")
                    print(str3)
                    process3 = subprocess.Popen(str3.strip(), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    # # 监控命令的运行状态和结果
                    stdout, stderr = process3.communicate()
                    exit_code = process.wait()
                    if exit_code != 0:
                        print("Cann't get bc of dynamic Library;")
                        print("error:" + command.strip())
                        print(stderr.decode())
                        sys.exit(0)
                    print("Get bc of dynamic Library!")
                    flag=1
                    break#一般只有一个
                elif len(parts) > 2 and "=>" in line and "/usr/local/lib" in line:
                    #检测到在/usr/local/lib
                    #print("4")
                    executable_path2 = parts[2]
                    output2 = subprocess.check_output(["file", executable_path2]).decode()
                    file_d_path = ""
                    if "symbolic link" in output2:
                        # ldd指向链接
                        #print("5")
                        parts2 = output2.split()
                        file_d = parts2[-1]
                        find_str=project_path+' -name '+file_d
                        #print(find_str)
                        try:
                            output3 = subprocess.check_output(["bash", "-c", "find "+find_str]).decode()
                        except subprocess.CalledProcessError:
                            #print("find command failed or didn't find any files")
                            output3 = ""
                        if output3 == "":
                            continue
                        file_d_path =output3
                    else:
                        #print("6")
                        parts2 = executable_path2.split("/")
                        file_d = parts2[-1]
                        find_str=project_path+" -name "+file_d
                        try:
                            output3 = subprocess.check_output(["find", find_str]).decode()
                        except subprocess.CalledProcessError:
                            #print("find command failed or didn't find any files")
                            output3 = ""
                        if output3 == "":
                            continue
                        file_d_path =output3
                    str3 = "extract-bc " + file_d_path.strip("\n")
                    print(str3)
                    process3 = subprocess.Popen(str3.strip(), shell=True, stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE)
                    # # 监控命令的运行状态和结果
                    stdout, stderr = process3.communicate()
                    exit_code = process.wait()
                    if exit_code != 0:
                        print("Cann't get bc of dynamic Library;")
                        print("error:" + command.strip())
                        print(stderr.decode())
                        sys.exit(0)
                    print("Get bc of dynamic Library!")
                    flag = 1
                    break  # 一般只有一个

        else:
            # 运行klee
            if os.path.exists(project_path + "/" + str(i) + "_ktest "):
                os.system("rm -rf "+project_path + "/" + str(i) + "_ktest")
            if flag==0:
                str1 = command.split(" ")
                num_cmd = len(str1)
                Running_instructions = " klee --max-time=14400 --search=dfs -optimize --libc=uclibc --posix-runtime " \
                                   "-only-output-states-covering-new " \
                                   "--output-dir=" + project_path + "/" + str(i) + "_ktest " + exe_path.strip("\n") + ".bc "
            else:
                str1 = command.split(" ")
                num_cmd = len(str1)
                Running_instructions = " klee --max-time=14400  --search=dfs -optimize --libc=uclibc --posix-runtime " \
                                       "-only-output-states-covering-new " \
                                       "--output-dir=" + project_path + "/" + str(i) + "_ktest " +" -link-llvm-lib="+file_d_path.strip("\n")+".bc "+ exe_path.strip(
                    "\n") + ".bc "
            for j in range(num_cmd):
                if j == 0:
                    continue
                # elif str1[j].strip() == "file":
                #     Running_instructions = Running_instructions + " A --sym-files 1 100 "
                else:
                    Running_instructions = Running_instructions + " " +str1[j].strip()
            print(Running_instructions)
            process1 = subprocess.Popen(Running_instructions.strip(), shell=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
            # 监控命令的运行状态和结果
            stdout1, stderr1 = process1.communicate()
            exit_code1 = process1.wait()
            if exit_code1 != 0:
                print("Klee failed to run;")
                print("error:" + command.strip())
                print(stderr1.decode())
                sys.exit(0)
            print("Get test case!")
            #得到replay
            #./replay.sh /home/klee/file/libtiff/libtiff-9/tools/fax2ps_ktest /home/klee/file/libtiff/libtiff-9/tools/fax2ps >
            #/home/klee/replay/libtiff-9-fax2ps-replay/replay.log
            replay_instructions = path + "/python-do/replay.sh " + project_path + "/" + str(i) + "_ktest "
            os.mkdir(replay_path + "/" + str1[0].strip())
            replay_instructions = replay_instructions + exe_path.strip("\n") + " > " + replay_path + "/" + str1[0].strip() + "/replay.log"
            print(replay_instructions)
            process2 = subprocess.Popen(replay_instructions.strip(), shell=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
            # 监控命令的运行状态和结果
            stdout2, stderr2 = process2.communicate()
            exit_code2 = process2.wait()
            if exit_code2 != 0:
                print("Failed to get dynamic information;")
                print("error:" + command.strip())
                print(stderr2.decode())
                sys.exit(0)
            print("Get dynamic information!")
