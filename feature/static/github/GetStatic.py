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
if os.path.exists("parse_result"):
    shutil.rmtree("parse_result")
if os.path.exists("raw_result"):
    shutil.rmtree("raw_result")
if os.path.exists("result"):
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
if os.path.exists(path+"/cfg_result"):
    shutil.rmtree(path+"/cfg_result")

'''
get  cfgpath and concate
'''

current_dir=os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir+"/joern-cli_new")
sliceDir=current_dir+"/cut/good"
cfg=current_dir+"/cfg"
cfg_result=current_dir+"/cfg_result"
sevenEdges=current_dir+"/static"
if os.path.exists(cfg):
    shutil.rmtree(cfg)
os.mkdir(cfg)
os.system("python3 getcfg.py "+sliceDir+" "+cfg)
os.chdir(current_dir+"/static_do/cfg2path/src")
os.system("java -classpath "+current_dir+"/static_do/cfg2path/out/production/cfg2path "+"cfg2path.GetCfgInfo "+cfg+" "+cfg_result)

os.chdir(current_dir)
os.system("python3 "+current_dir+"/check_cfg.py "+current_dir+"/cfg_result "+current_dir+"/cfg_result_exception")

os.chdir(current_dir+"/static_do/cfg2path/src")
os.system("java -classpath "+current_dir+"/static_do/cfg2path/out/production/cfg2path "+"cfg2path.Concate2File "+cfg_result+" "+sevenEdges)
