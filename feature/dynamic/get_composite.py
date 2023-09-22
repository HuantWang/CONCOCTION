import os
import subprocess
import shutil
import sys

# #path=os.getcwd()
path = os.path.dirname(os.path.realpath(__file__))
allreplay_dir = path+"/dynamic"
replay_lists=os.listdir(allreplay_dir)
for replay_list in replay_lists:
    replay_dir=allreplay_dir+"/"+replay_list
    if os.path.exists(replay_dir+"/static"):
        shutil.rmtree(replay_dir+"/static")
    #static和dynamic要在同一目录下
    shutil.copytree("/home/feature/static/github/static", replay_dir+"/static")
    os.chdir(path+"/python-do")

#    # 执行 cut.py 脚本
    #print(replay_dir)
    os.system("python3 ./cut.py "+replay_dir)

#     # 将 n.txt 存入 cut 文件夹
    if os.path.exists(replay_dir+"/cut"):
        shutil.rmtree(replay_dir+"/cut")
    os.makedirs(os.path.join(replay_dir, "cut"), exist_ok=True)
    #print("mv "+replay_dir+"/*.txt"+os.path.join(replay_dir, "cut"))
    os.system("mv "+replay_dir+"/*.txt "+os.path.join(replay_dir, "cut"))

#     # 建立 result 文件夹，并将静态特征复制到 result 文件夹
    if os.path.exists(replay_dir+"/result"):
        shutil.rmtree(replay_dir+"/result")
    os.makedirs(os.path.join(replay_dir, "result"), exist_ok=True)
    shutil.copytree(os.path.join(replay_dir, "static"), os.path.join(replay_dir, "result/static"))

#     # 执行 automated_good.py 脚本
#     #print("python3 /home/feature/dynamic/python-do/automated.py "+replay_dir)
    os.system("python3 "+path+"/python-do/automated.py "+replay_dir)

#返回 dynamic 目录
os.chdir(path)
print("Get composite features!")

#将所有特征放到一个文件夹中
if os.path.exists(path+"/composite"):
    shutil.rmtree(path+"/composite")
#os.makedirs(path+"/composite", exist_ok=True)
replay_lists=os.listdir(allreplay_dir)
for i,replay_list_gather in enumerate(replay_lists):
    replay_dir_gather = os.path.join(allreplay_dir, replay_list_gather)
    shutil.copytree(os.path.join(replay_dir_gather, "composite"),path+"/composite")
    file_lists=os.listdir(path+"/composite")
    for file_list in file_lists:
        #改名，一个项目多个应用会导致函数重名
        os.rename(os.path.join(path+"/composite",file_list),os.path.join(path+"/composite",str(i)+file_list))
