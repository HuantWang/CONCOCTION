import subprocess
import os
from shutil import copyfile
import time,sys
def process(filepath,storedpath):
    cwdir=r"/home/feature/static/github/joern-cli_new"


    path1="1cpgbin.bin"
    if os.path.exists(path1):
        os.remove(path1)
    path2 = "1cfg.txt"
    path3 = "1astNode.txt"
    print(filepath)
    s=subprocess.Popen("sh joern-parse "+filepath+" -o "+path1,shell=True,cwd=cwdir)
    s.wait(20)


    p=subprocess.Popen("sh joern",shell=True,stdin=subprocess.PIPE,encoding="utf-8",cwd=cwdir)

    loadcmd="loadCpg(\""+path1+"\")\n"
    cfgcmd = f"cpg.method.filter(_.isExternal == false).dotCfg.l|>\"{path2}\"\n"
    astNodecmd=f"cpg.method.filter(_.isExternal == false).ast.code.l|>\"{path3}\"\n"
    data=p.communicate(input=loadcmd+cfgcmd+astNodecmd,timeout=60)
    #print(data)
    try:
        p.wait(50)
    except:
        p.kill()
    copyfile(path2,storedpath+"-cfg.txt")
    copyfile(path3,storedpath+"-ast.txt")
    time.sleep(2)


def getList(dir):
    fileName = os.listdir(dir)
    return fileName

if __name__=='__main__':
    if len(sys.argv) < 3 or len(sys.argv)>3:
        print("请分别输入切片后的函数所在文件夹,中间文件存储位置:")
        print("eg: python getcfg.py E:\\tt\\badd E:\\tt\\badd_result")
        sys.exit()
    elif len(sys.argv) == 3:
        cfiledir = sys.argv[1]
        storedir = sys.argv[2]


    # cfiledir=r"E:\tt\badd"
    # storedir=r"E:\tt\badd_result"
    if not os.path.exists(storedir):
        os.mkdir(storedir)

    cfileName=getList(cfiledir)
    cfileName.sort()
    for i in cfileName:
        storedpath=storedir+"/"+i
        print(i)
        process(cfiledir+"/"+i,storedpath)
