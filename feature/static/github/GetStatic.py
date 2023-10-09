import os
import shutil
import sys
import logging


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(message)s',
                     level=logging.INFO)
def log(mess):
     logging.info(mess)
     
project_path=sys.argv[1]
log(f'project : {project_path}')
path = os.path.dirname(os.path.realpath(__file__))
logDir=path+"/log"
if not os.path.exists(logDir):
    os.mkdir(logDir)

if os.path.exists(path+"/cut"):
    shutil.rmtree(path+"/cut")
if os.path.exists(path+"/out"):
    shutil.rmtree(path+"/out")
if os.path.exists(path+"/static"):
    shutil.rmtree(path+"/static")
if not os.path.exists(project_path):
    log(f"project {project_path} not exist!")

log("start to get function segmentation... ")
os.chdir(path+"/static_do/copy/src")
os.system("java ClassifyFileOfProject.java "+project_path+" "+path)
os.chdir(path+"/static_do/cut/src/slice")
os.system("java -classpath "+path+"/static_do/cut/out/production/cut:"+path+"/static_do/cut/lib/cdt.core-5.6.0.201402142303.jar:"+path+"/static_do/cut/lib/equinox.common-3.6.200.v20130402-1505.jar slice.Main "+path)

log("start to get graphRelation by joern... ")
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

log("start to get seven edges...")
os.chdir(path+"/static_do/side/src/sevenEdges/")
os.system("java -classpath "+path+"/static_do/side/out/production/side:"+path+"/static_do/side/lib/cdt.core-5.6.0.201402142303.jar:"+path+"/static_do/side/lib/equinox.common-3.6.200.v20130402-1505.jar sevenEdges.Main "+path+" >"+logDir+"/1sevenEdges.Main.log")
os.system("java -classpath "+path+"/static_do/side/out/production/side:"+path+"/static_do/side/lib/cdt.core-5.6.0.201402142303.jar:"+path+"/static_do/side/lib/equinox.common-3.6.200.v20130402-1505.jar sevenEdges.concateJoern "+path+" >"+logDir+"/2sevenEdges.concateJoern.log")



if os.path.exists(path+"/cfg"):
    shutil.rmtree(path+"/cfg")
if os.path.exists(path+"/cfg_result"):
    shutil.rmtree(path+"/cfg_result")

log("start to get cfgpath  by joern and concate... ")
current_dir=path
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
os.system("java -classpath "+current_dir+"/static_do/cfg2path/out/production/cfg2path "+"cfg2path.GetCfgInfo "+cfg+" "+cfg_result+" >"+logDir+"/3cfg2path.GetCfgInfo.log")

os.chdir(current_dir)
os.system("python3 "+current_dir+"/check_cfg.py "+current_dir+"/cfg_result "+current_dir+"/cfg_result_exception")

os.chdir(current_dir+"/static_do/cfg2path/src")
os.system("java -classpath "+current_dir+"/static_do/cfg2path/out/production/cfg2path "+"cfg2path.Concate2File "+cfg_result+" "+sevenEdges+" >"+logDir+"/4cfg2path.Concate2File.log")

print("============================================================================================================")
log(f"extracted static feature stored in : {path}/static/")