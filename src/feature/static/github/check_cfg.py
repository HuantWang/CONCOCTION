import os
import sys
import shutil
# 指定要检查的目录
dir_path = sys.argv[1]
path_mv= sys.argv[2]

if not os.path.exists(dir_path):
    os.makedirs(path_mv)

# 遍历目录下的所有文件
for filename in os.listdir(dir_path):
    # 检查是否是 txt 文件
    if filename.endswith('.txt'):
        file_path = os.path.join(dir_path, filename)
        # 读取文件内容
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 检查最后一行是否为 "====================================="
        if lines[-1].strip() != '=====================================':
            # 如果不是，则删除文件
            shutil.move(file_path,path_mv)

            # 输出文件名
            print('cfg information doesn\'t exist:'+file_path)
