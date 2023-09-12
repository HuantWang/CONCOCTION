import json
import os

datas=[]

def extract(rootpath):
    filelist=os.listdir(rootpath)
    index = 0
    for item in filelist:
        if os.path.isfile(rootpath+'\\'+item):
            with open(rootpath + '\\' + item,"r")as file_1:
                flag=0
                features={}
                line=file_1.readline()
                codes=""
                while line.find("---children---") == -1:
                    if line.find("-----label-----")==-1 and flag==0:
                        features['project'] = "Jasper"
                        features['commit_id']= '973b1a6b9070e2bf17d17568cbaf4043ce931f51'
                        features['target']=int(line)
                        flag=1
                        line = file_1.readline()
                    elif line.find("-----code-----")==-1 and flag==1:
                        codes+=line
                        line = file_1.readline()
                    else:
                        line = file_1.readline()
                flag=0
                features['func']=codes
                index+=1
                datas.append(features)
    with open('dataset.json','w') as file:
        file.write(json.dumps(datas,indent = 2))
    print(datas)
rootpath=r"C:\Users\MSI\Desktop\funded_test"
extract(rootpath)