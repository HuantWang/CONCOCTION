import os
import shutil
import sys

#输入参数 ：replay.log所在目录，如 /xxx/yyy/replay/replay.log，则输入/xxx/yyy/replay
#被automated_good.py调用执行
#将动态特征和静态特征存入good

# final_path=r'/home/klee/new/ok-file-formats/ok-file-formats_f/ok-file-formats_f_backup/result/dynamic'
folder_name = sys.argv[1]
# filePath=r"/home/klee/new/ok-file-formats/ok-file-formats_f/ok-file-formats_f_backup/"+folder_name+"/"
path_pwd = sys.argv[2]
concat_file=path_pwd +"/composite/"
#concat_file=r"/xxx/yyy/replay/good/"

if not os.path.isdir(concat_file):
	os.makedirs(concat_file)

def get(name, info):
    
    str=concat_file+"{}"
    #/xxx/yyy/replay/good/{}
    with open(str.format(folder_name+name), "a") as f:
        f.write("".join(info))


path=path_pwd+f"/result"
#path = r"/xxx/yyy/replay/result"
lists = os.listdir(path)
# shutil.copy(filePath+'*',final_path)
for l in lists:
    
    p = os.path.join(path, l)
    ps = os.listdir(p)
    for p1 in ps:
        
        p2 = os.path.join(p, p1)
        
        with open(p2, "r") as f:
            lines = f.readlines()
            get(p1, lines)
