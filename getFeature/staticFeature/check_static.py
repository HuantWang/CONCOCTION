import os
import sys
import shutil

path_src=sys.argv[1]
path_all= path_src
path_mv= path_src+"_nocfg"

if not os.path.exists(path_mv):
    os.makedirs(path_mv)

for path in os.listdir(path_all):
    path=path_all+"/"+path
    with open(path, 'rb') as file:
        # 将文件指针移动到文件末尾
        file.seek(0, 2)
        # 获取文件末尾位置
        end_position = file.tell()
        # 初始化位置和字符变量
        position = 1
        last_line = ""
        while position <= end_position:
            file.seek(-position, 2)
            character = file.read(1)
            if character == b"\n":
                if last_line:
                    second_last_line = file.readline().decode().strip()
                    break
                else:
                    last_line = file.readline().decode().strip()
            position += 1
        # 打印最后一行内容
        #print(last_line)
        if last_line == "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^":
            print(path)
            shutil.move(path,path_mv)
            

