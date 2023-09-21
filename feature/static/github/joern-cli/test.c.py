import subprocess
import os
import sys
spath = sys.argv[1]
str1 = "sh joern-parse " + sys.argv[1] + "/cut/good" + " --out parse_result/good.bin"
s = subprocess.Popen(str1, shell=True)
s.wait(1000)
s.kill()
cmd = ["sh", "joern"]

#   启动进程
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, encoding="utf-8")

# 向进程写入命令并等待
p.stdin.write("loadCpg(\"/home/feature/static/github/joern-cli/parse_result/good.bin\")\n")

# 再次向进程写入命令并等待
p.stdin.write("cpg.runScript(\"/home/feature/static/github/joern-cli/graph/allgood.sc\")\n")
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



