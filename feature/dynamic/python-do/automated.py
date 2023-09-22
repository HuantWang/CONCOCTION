import os
import sys
#输入参数 ：replay.log所在目录，如 /xxx/yyy/replay/replay.log，则输入/xxx/yyy/replay
#被do.sh调用执行
#获取good的混合特征
str1=sys.argv[1]
path1=str1+f"/cut"
#path1=/xxx/yyy/replay/cut
fileList=os.listdir(path1)

num_file = len(fileList)
print(num_file)

# os.system("python3 ./cut.py")

for i in range(1,num_file+1):
    print(i)
    if i > 0 :
        #os.system("python3 ./replenish_end.py "+str(i)+" "+str1)
        os.system("python3 ./extract_file-func.py "+str(i)+" "+str1)
        os.system("python3 ./insert_testcase.py "+str(i)+" "+str1)
        os.system("python3 ./before_concat.py "+str(i)+" "+str1)
        os.system("python3 ./concat.py "+str(i)+" "+str1)#将静态和动态混合
        print("==========")

os.system("python3 ./after_concat.py"+" "+str1)#删除只有静态的
os.system("python3 ./delet_dynamicfile.py"+" "+str1)#删除只有动态的

