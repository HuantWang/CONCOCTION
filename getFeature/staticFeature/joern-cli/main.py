import subprocess
import os
import sys
spath = sys.argv[1]
str1 = "sh joern-parse " + sys.argv[1] + "/cut/good" + " --out parse_result/good.bin"
s = subprocess.Popen(str1, shell=True)
s.wait(1000)
s.kill()
cmd = ["sh", "joern"]


p = subprocess.Popen(cmd, stdin=subprocess.PIPE, encoding="utf-8")

binPath=os.path.join(os.path.dirname(os.path.realpath(__file__)),"parse_result/good.bin")
scPath=os.path.join(os.path.dirname(os.path.realpath(__file__)),"graph/allgood.sc")
cmd1="loadCpg(\""+binPath+"\")\n"
cmd2="cpg.runScript(\""+scPath+"\")\n"
p.stdin.write(cmd1)
p.stdin.write(cmd2)
p.stdin.flush()
try:
    p.wait(500)
except:
    p.kill()

e = subprocess.Popen("python3 joern_relationgood.py", shell=True)
try:
    e.wait(60)
except:
    e.kill()



