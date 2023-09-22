import os
import shutil
import sys

#输入参数 ：replay.log所在目录，如 /xxx/yyy/replay/replay.log，则输入/xxx/yyy/replay
#被automated_good.py/automated_bad.py调用执行
#将动态特征放入dynamic
folder_name = sys.argv[1]
path_pwd=sys.argv[2]
src_path=path_pwd+"/testcase-trace/"+folder_name+"/"
#src_path=r"/xxx/yyy/replay/testcase-trace/"+folder_name+"/"

target_path=path_pwd+"/result/dynamic/"
#target_path=r'/xxx/yyy/replay/result/dynamic/'

path=path_pwd+"/result/dynamic"
#path=r'/xxx/yyy/replay/result/dynamic'

if os.path.exists(path):
    shutil.rmtree(path)

if not os.path.isdir(target_path):      
		os.makedirs(target_path)

file_list=os.listdir(src_path)
if len(file_list)>0:
    for file in file_list:
        shutil.copy(src_path+file,target_path+file)
