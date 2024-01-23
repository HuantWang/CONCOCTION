import subprocess
import os
from shutil import copyfile
import time,sys
import logging
from tqdm import tqdm

logging.basicConfig(filename='getcfg.py.log',
                     format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                     level=logging.INFO)
def log_text(p,cmd):
    out,err = p.communicate()
    logging.info('cmd: '+cmd)
    logging.info('returncode: ' + str(p.returncode))
    logging.info('stdout:')
    logging.info(out)
    logging.info('stderr:')
    logging.info(err)
    
def process(filepath,storedpath):
    # cwdir=r"/home/feature/static/github/joern-cli_new"
    cwdir = os.path.dirname(os.path.realpath(__file__))


    path1="1cpgbin.bin"
    if os.path.exists(path1):
        os.remove(path1)
    path2 = "1cfg.txt"
    path3 = "1astNode.txt"
    cmd="sh joern-parse "+filepath+" -o "+path1
    s=subprocess.Popen(cmd,shell=True,cwd=cwdir,
                         stdout = subprocess.PIPE,
                         stderr = subprocess.PIPE,
                         universal_newlines=True,
                         bufsize = 1)
    log_text(s,cmd)
    s.wait(20)
    

    cmd1="sh joern"
    p=subprocess.Popen(cmd1,shell=True,stdin=subprocess.PIPE,encoding="utf-8",cwd=cwdir,stdout = subprocess.PIPE,
                         stderr = subprocess.PIPE,
                         universal_newlines=True,
                         bufsize = 1)

    loadcmd="loadCpg(\""+path1+"\")\n"
    cfgcmd = f"cpg.method.filter(_.isExternal == false).dotCfg.l|>\"{path2}\"\n"
    astNodecmd=f"cpg.method.filter(_.isExternal == false).ast.code.l|>\"{path3}\"\n"
    # data=p.communicate(input=loadcmd+cfgcmd+astNodecmd,timeout=60)
    p.stdin.write(loadcmd)
    p.stdin.write(cfgcmd)
    p.stdin.write(astNodecmd)
    p.stdin.write('exit\n')
    p.stdin.flush()
    li=[cmd1,loadcmd,cfgcmd,astNodecmd]
    log_text(p,str(li))

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
    filebar=tqdm(cfileName)
    for i in filebar:
        filebar.set_description('Processing getcfg '+i)
        storedpath=storedir+"/"+i
        process(cfiledir+"/"+i,storedpath)
    # for i in cfileName:
    #     storedpath=storedir+"/"+i
    #     process(cfiledir+"/"+i,storedpath)
