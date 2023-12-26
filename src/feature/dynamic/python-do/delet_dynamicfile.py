import os
import string
from shutil import copy
import shutil
import sys

#输入参数 ：replay.log所在目录，如 /xxx/yyy/replay/replay.log，则输入/xxx/yyy/replay
#被automated_good.py调用执行
#删除没有静态的特征
first='-----label-----\n'
last='=======================\n'     

path_pwd=sys.argv[1]
str=path_pwd+"/composite/"
for root,dirs,files in os.walk(str):
    for file in files:
        
        pre_path = os.path.join(root,file)
        # print(pre_path)
        filename = os.path.split(pre_path)[1]
        # print(filename)
        with open(pre_path, 'r') as fp:
            lines = fp.readlines()
            first_line = lines[0]
            # print(first_line)
            last_line = lines[-1]
            # print(last_line)
            if first_line != first :
            # if last_line != last or first_line != first:
                # print("------------------------------------------")
                os.remove(pre_path)
            if last_line != last :
            # elif first_line != first :
                os.remove(pre_path)
                # copy(pre_path, to_path)
            else :
                continue
