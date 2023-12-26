import os
current=os.path.dirname(__file__)

fp=current+"/setRootpath.log"
def setRootpath():
    ROOTPATH=os.getcwd()
    print(f"ROOTPATH:{ROOTPATH}")
    
    with open(fp,"w") as f:
        f.write(ROOTPATH)
    return ROOTPATH
        
def getRootpath():
    with open(fp,"r") as f:
        Rootpath=f.readline()
        return Rootpath
    

