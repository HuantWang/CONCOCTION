import os
import shutil
import sys
project_path=sys.argv[1]
print(project_path)
path = os.path.dirname(os.path.realpath(__file__))
print(path)
if os.path.exists(path+"/cut"):
    shutil.rmtree(path+"/cut")
if os.path.exists(path+"/out"):
    shutil.rmtree(path+"/out")
if os.path.exists(path+"/static"):
    shutil.rmtree(path+"/static")
if not os.path.exists(project_path):
    print("File not exist!")
os.chdir(path+"/static_do/copy/src")
os.system("java ClassifyFileOfProject.java "+project_path+" "+path)

os.chdir(path+"/static_do/cut/src/slice")
os.system("java -classpath "+path+"/static_do/cut/out/production/cut:"+path+"/static_do/cut/lib/cdt.core-5.6.0.201402142303.jar:"+path+"/static_do/cut/lib/equinox.common-3.6.200.v20130402-1505.jar slice.Main "+path)
print("Get function segmentation ! ")


os.chdir(path+"/joern-cli")
shutil.rmtree("parse_result")
shutil.rmtree("raw_result")
shutil.rmtree("result")
os.mkdir("parse_result")
os.mkdir("raw_result")
os.mkdir("result")
os.system("python3 main.py "+path)
os.chdir(path+"/static_do/side/src/sevenEdges/")
os.system("java -classpath "+path+"/static_do/side/out/production/side:"+path+"/static_do/side/lib/cdt.core-5.6.0.201402142303.jar:"+path+"/static_do/side/lib/equinox.common-3.6.200.v20130402-1505.jar sevenEdges.Main "+path)
os.system("java -classpath "+path+"/static_do/side/out/production/side:"+path+"/static_do/side/lib/cdt.core-5.6.0.201402142303.jar:"+path+"/static_do/side/lib/equinox.common-3.6.200.v20130402-1505.jar sevenEdges.concateJoern "+path)
print("Get seven edges!")

if os.path.exists(path+"/cfg"):
    shutil.rmtree(path+"/cfg")
if os.path.exists(path+"/cfgresult"):
    shutil.rmtree(path+"/cfgresult")
if os.path.exists(path+"/cfgresult1"):
    shutil.rmtree(path+"/cfgresult1")
if os.path.exists(path+"/cfgresult_bad"):
    shutil.rmtree(path+"/cfgresult_bad")
os.chdir(path+"/joern-cli_new")

os.system("python3 "+path+"/joern-cli_new/getcfg.py "+path+"/cut/good "+path+"/cfg")
print("Get cfg !")
os.mkdir(path+"/cfg1")
os.mkdir(path+"/cfgresult_bad")
if os.path.exists(path+"/test1"):
    shutil.rmtree(path+"/test1")
os.mkdir(path + "/test1")

while os.listdir(path+"/cfg"):

    cfg_files = os.listdir(path+"/cfg")[:2]
    for filename in cfg_files:
        os.rename(os.path.join(path+"/cfg", filename), os.path.join(path+"/test1", filename))


    os.chdir(path+"/static_do/cfg2path/src")
    os.system(
        "java -classpath "+path+"/static_do/cfg2path/out/production/cfg2path cfg2path.GetCfgInfo /home/test1 "+path+"/cfgresult")


    test1_files = os.listdir(path+"/test1")
    for filename in test1_files:
        os.rename(os.path.join(path+"/test1", filename), os.path.join(path+"/cfg1", filename))

    os.chdir(path)
print("Cfgresult completed")
shutil.copytree(path+"/cfgresult", path+"/cfgresult1")

os.chdir(path)
os.system("python3 ./check_cfg.py "+path+"/cfgresult "+path+"/cfgresult_bad")

os.chdir(path+"/static_do/cfg2path/src")
os.system("java -classpath "+path+"/static_do/cfg2path/out/production/cfg2path cfg2path.Concate2File "+path+"/cfgresult "+path+"/static")


os.chdir(path)
os.system("python3 check_static.py " + path+"/static")
print("Get static feature!")