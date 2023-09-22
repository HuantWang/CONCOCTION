import os
import re
import sys

#输入参数 ：replay.log所在目录，如 /xxx/yyy/replay/replay.log，则输入/xxx/yyy/replay
#被automated_good.py调用执行
#对所有切片插入testcase

lines=[]
regex3 = r"(?<=object [0-9]: name: ').+(?=')"
regex4 = r"(?<=object [0-9]: hex : ).+"

folder_name = sys.argv[1]
path=sys.argv[2]
filePath=path+"/testcase-trace"+"/"+folder_name+"/"
#filePath=r"/xxx/yyy/replay/testcase-trace/"+folder_name+"/"


fileList=os.listdir(filePath)


#cut_files = /xxx/yyy/replay/cut/1.txt
with open(path+'/cut/'+folder_name+'.txt', 'r') as f:
# with open('/xxx/yyy/replay/cut/6.txt', 'r') as f:
    test_str = f.read()

matches3 = re.finditer(regex3, test_str, re.MULTILINE)
matches4 = re.finditer(regex4, test_str, re.MULTILINE)

for matchNum3, match3 in enumerate(matches3, start=1):
    for matchNum4, match4 in enumerate(matches4, start=1):
        for file in fileList:
            f=open(os.path.join(filePath,file),'r')
            for line in f:
                lines.append(line)
            f.close()
            lines.insert(2,match3.group()+":"+match4.group()+"\n")
            s=''.join(lines)
            f=open(os.path.join(filePath,file),'w+') 
            f.write(s)
            f.close()
            del lines[:]
            continue    
        break
