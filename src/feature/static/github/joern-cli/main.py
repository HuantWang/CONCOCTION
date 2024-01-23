import subprocess
import os
import sys
import logging

logging.basicConfig(filename='main.py.log',
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

    
spath = sys.argv[1]
str1 = "sh joern-parse " + sys.argv[1] + "/cut/good" + " --out parse_result/good.bin"
s = subprocess.Popen(str1, shell=True,stdout = subprocess.PIPE,
                         stderr = subprocess.PIPE,
                         universal_newlines=True,
                         bufsize = 1)
s.wait(1000)
s.kill()
log_text(s,str1)

cmd = ["sh", "joern"]


p = subprocess.Popen(cmd, stdin=subprocess.PIPE, encoding="utf-8",
                         stdout = subprocess.PIPE,
                         stderr = subprocess.PIPE,
                         universal_newlines=True,
                         bufsize = 1)  

binPath=os.path.join(os.path.dirname(os.path.realpath(__file__)),"parse_result/good.bin")
scPath=os.path.join(os.path.dirname(os.path.realpath(__file__)),"graph/allgood.sc")
cmd1="loadCpg(\""+binPath+"\")\n"
cmd2="cpg.runScript(\""+scPath+"\")\n"
p.stdin.write(cmd1)
p.stdin.write(cmd2)
p.stdin.write('exit\n')
p.stdin.flush()
cmd.append(cmd1)
cmd.append(cmd2)
log_text(p,str(cmd))
try:
    p.wait(500)
except:
    p.kill()



e = subprocess.Popen("python3 joern_relationgood.py", shell=True)
try:
    e.wait(60)
except:
    e.kill()



