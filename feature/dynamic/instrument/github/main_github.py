#用于github项目插桩

import os, sys,subprocess,os.path
import datetime
from pathlib import Path
import shutil
import argparse,re
instrument_dir=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
def copyDirecFile(source_path,target_path):
    if os.path.exists(source_path):
        shutil.copytree(source_path, target_path)


    print('copy directory files finished!')

def getFile(projectPath):
    li={}
    for root,folder_names, file_names in os.walk(projectPath):
        for file_name in file_names :
            if file_name.endswith(".c") or file_name.endswith(".cpp"):
                li[file_name]=root+"/"+file_name
    return li

def getDir(projectPath):
    li=[]
    for dirPath, t, filenameList in os.walk(projectPath):
        for item in filenameList:
            if item.find(".h")!=-1:
                if li.count(dirPath)==0:
                    li.append(dirPath)
            if dirPath.endswith("/include"):
                if li.count(dirPath)==0:
                    li.append(dirPath)
    return li

def getcmd(filePathList,includePath,flag,logPath):
    includep=''
    cmd= {}
    if includePath is not None:
        for item in includePath:
            includep=includep+" -I"+item
    #a 为加括号的可执行文件地址/插桩可执行文件地址
    if flag==1:
        a=instrument_dir+"/clangTool/cmake-build-debug/bin/clang-addBrace "
        for name,path in filePathList.items():
            # cmd[name]=(a+path+" --"+includep+">"+logPath+"/"+name+"_addbrace.log")
            cmd[name]=(a+path+" --"+includep+" 2>"+logPath+"/"+name+"_addbrace.log")

    if flag==2:
        a=instrument_dir+"/clangTool/cmake-build-debug/bin/clang-diff "
        for name,path in filePathList.items():
            cmd[name]=(a+path+" --"+includep+" 2>"+logPath+"/"+name+"_insertPrint.log")
    return cmd








def execAddBraceCmd(cmd):
    try:
        os.mkdir(os.getcwd() + '/execAddBraceCmd')
    except:
        print("os.getcwd() + '/execAddBraceCmd' exist!")
    for fileName,cmd in cmd.items():
        print("AddBraceCmd current cmd:")
        print(cmd)
        child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        while True:
            if child.poll!=None:
                break
        if child.poll()==1:
            child.terminate()
        file = open(os.getcwd() + '/execAddBraceCmd/' + fileName + '_addBrace.txt', 'w+', encoding='utf-8')
        file.writelines(cmd+'\n')
        file.write(str(child.stdout.read()))
        file.write(str(child.stderr.read()))
        file.close()

    # proc = subprocess.Popen(["pgrep", "gedit"], stdout=subprocess.PIPE)
    # for pid in proc.stdout:
    #     os.kill(int(pid), signal.SIGTERM)




def execInsertPrintCmd(cmd):
    try:
        os.mkdir(os.getcwd() + '/execInsertPrintCmd')
    except:
        print("os.getcwd() + '/execInsertPrintCmd' exist!")
    for fileName,cmd in cmd.items():
        print("InsertPrintCmd current cmd:")
        print(cmd)
        child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        while True:
            if child.poll!=None:
                break
        if child.poll()==1:
            child.terminate()
        file = open(os.getcwd() + '/execInsertPrintCmd/' + fileName + '_insertPrint.txt', 'w+', encoding='utf-8')
        file.writelines(cmd+'\n')
        file.write(str(child.stdout.read()))
        file.write(str(child.stderr.read()))
        file.close()



def annotationAdd(filePath):
    regx = re.compile("#ifdef .*")
    fileIn=filePath
    fileOut=filePath+"_tmp.c"
    with open(fileIn, 'r') as fpIn, open(fileOut, 'w') as fpOut:
        for line in fpIn:
            if re.match(regx,line):
                fpOut.write("#define "+line[7:].rstrip()+" 1 //annotation0569 \n")
            fpOut.write(line)
    os.remove(filePath)
    os.rename(fileOut,filePath)
    print(f"annotationAdd for {filePath}")



def annotationDelet(filePath):
    regx = re.compile("#define .*(//annotation0569)")
    fileIn=filePath
    fileOut=filePath+"_tmp.c"
    with open(fileIn, 'r') as fpIn, open(fileOut, 'w') as fpOut:
        for line in fpIn:
            if re.match(regx,line)==None:
                fpOut.write(line)
    os.remove(filePath)
    os.rename(fileOut,filePath)
    print(f"annotationDelet for {filePath}")






if __name__ == '__main__':
    # python3.7 main.py --path /home/libpng-1.2.7 -I/home/libzip-1-2_backup_Backup /home/insertlibzip
    parser = argparse.ArgumentParser()
    parser.add_argument("-path",help="project path")
    parser.add_argument("-logDir",help="stored log path")
    parser.add_argument("-I",nargs='*',help="head file address list")

    args = parser.parse_args()
    print(f"project path:{args.path}")
    print(f"include path:{args.I}")



    includePath=args.I
    projectPath=args.path
    filePathList=getFile(projectPath)#获取所有*.c *.cpp文件地址
    try:
        includePath.extend(getDir(projectPath))#将项目中所有添加.h文件的路径加入到includePath列表中
    except:
        includePath=getDir(projectPath)
    print(includePath)



    logPath1=args.logDir + '/insertlog/execAddBraceCmd'
    logPath2=args.logDir + '/insertlog/execInsertPrintCmd'
    if os.path.exists(args.logDir)==False:
        os.mkdir(args.logDir)
    try:
        os.mkdir(args.logDir+"/insertlog")
    except:
        print(f"mkdir {args.logDir}/insertlog failed!")
        sys.exit()

    os.mkdir(logPath1)
    os.mkdir(logPath2)

    copyDirecFile(projectPath,projectPath+"_backup")#将当前文件备份

#给所有c文件#ifdef xx的变量均 加定义以及注释#defin xx 1 //annotation0569
    for cfile in filePathList:
        annotationAdd(filePathList[cfile])
    print("annotationAdd end"+"*"*100)

#将加大括号(step1)以及插桩(step2)命令写入shell文件
    CMD1=getcmd(filePathList,includePath,1,logPath1)
    file1 = open(logPath1+'/' +  '0addBraceCMD.sh', 'w+', encoding='utf-8')
    for name,cmd in CMD1.items():
        file1.writelines("filename=\""+name+"\"\n")
        file1.writelines(cmd+'\n')
        file1.writelines("echo \"addbrace end for $filename \"\n")
    file1.close()


    CMD2=getcmd(filePathList,includePath,2,logPath2)
    file2 = open(logPath2+'/' +  '0insertPrintCMD.sh', 'w+', encoding='utf-8')
    for name,cmd in CMD2.items():
        file2.writelines("filename=\""+name+"\"\n")
        file2.writelines(cmd+'\n')
        file2.writelines("echo \"=====================================insertprint end for $filename \"\n")
    file2.close()

#分别执行加大括号以及插桩shell文件
    child = subprocess.Popen("sh "+logPath1+'/'+'0addBraceCMD.sh', shell=True)
    child.wait()
    child2 = subprocess.Popen("sh "+logPath2+'/'+'0insertPrintCMD.sh',shell=True)
    child2.wait()

#给所有c文件#defin xx 1的变量且包含注释//annotation0569 删除
    for cfile in filePathList:
        annotationDelet(filePathList[cfile])
    print("annotationDelete end"+"*"*100)




