import re
import sys

#输入参数 ：replay.log所在目录，如 /xxx/yyy/replay/replay.log，则输入/xxx/yyy/replay
#被do.sh调用执行
#对replay.log按testcase切片
regex = r"=======testcase========\n(.*\n)+?======================="
str1=sys.argv[1]
path1=str1+f"/replay.log"
#path1=/xxx/yyy/replay/replay.log
with open(path1, 'r') as f:
    test_str = f.read()
matches = re.finditer(regex, test_str, re.MULTILINE)

for matchNum0, match0 in enumerate(matches, start=1):
    str2=f"/{matchNum0}.txt"
    path=str1+str2
    #path=/xxx/yyy/replay/n.txt
    with open(path, "w", encoding='utf-8') as f:
        f.write(match0.group())
