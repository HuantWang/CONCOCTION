import os
import time
import random

def addTime(Dname):
    fileList = os.listdir(Dname)
    #    print(fileList)
    for item in fileList:
        # 如果是文件，就直接重命名
        if os.path.isfile(Dname + '\\' + item):
            with open(Dname + '\\' + item,"r")as file_1,open(r'C:\Users\MSI\Desktop\test2' + '\\' + "fact_"+str(random.random())+item, 'w') as file_2:
                line = file_1.readline()
                while line != '' and  line.find("-----dynamic-----") == -1:
                    if (line.find("-----dynamic-----") == -1):
                        file_2.writelines(line)
                    line = file_1.readline()
            # i = item.split('.')
            # newName = time.strftime('%Y-%m-%d', time.localtime()) + '_' + i[0] + '.' + i[1]+str(time.time())+".txt"

            # print(newName)
            # print(os.getcwd())
            # 如果要重命名，必须回到文件的当前目录，不然会报错，找不到文件
            os.chdir(Dname)
            # os.rename(item, newName)
        # 如果是目录，递归操作
        elif os.path.isdir(Dname + '\\' + item):
            print(item)
            addTime(Dname + '\\' + item)
        pass
    pass


Dname = r"C:\Users\MSI\Desktop\exms"
addTime(Dname)
