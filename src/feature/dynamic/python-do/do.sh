#!/bin/bash
#输入参数 ：replay.log所在目录，如 /xxx/yyy/replay/replay.log，则输入/xxx/yyy/replay
#要求在/xxx/yyy/replay下有文件夹static，static中有goodseven文件夹，文件夹内存有good的静态特征；有baddseven文件夹，文件夹内存有bad的静态特征
#得到n.txt
mv /home/feature/static/github/static /home/feature/dynamic/dynamic/
cd /home/feature/dynamic/python-do
python3 /home/feature/dynamic/python-do/cut.py $1
#将n.txt存入cut
sudo mkdir $1/cut
sudo mv $1/*.txt $1/cut
#建立result，存入good静态
sudo mkdir $1/result
sudo mkdir $1/result/dynamic
sudo cp -r $1/static/goodseven $1/result/good
#执行
sudo python3 /home/feature/dynamic/python-do/automated_good.py $1
cd /home/feature/dynamic
