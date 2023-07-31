# import os
import shutil

featurePath = "../SentEval/data/sard/good_416"
for root, dirs, files in os.walk("../SentEval/data/sard/cwe-416/good"):
    for file in files:
        # 获取文件所属目录
        # print("root", root)
        predir = os.path.join(root, file)
        # print("predir", predir)

        # 子文件夹下的文件名字
        fileList = os.listdir(root)
        # print("fileList", fileList)
        test = str(fileList[0]).endswith(".txt")
        if test == True:
            sample = fileList[0]  # 随机选取picknumber数量的样本图片

            # print("sample", sample)

            # for name in sample:

            pre_copy = root + "/" + sample
            # print(pre_copy)
            after_copy = featurePath + "/" + sample
            print(after_copy)
            shutil.copy(pre_copy, after_copy)
