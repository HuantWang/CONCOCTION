import matplotlib.pyplot as plt
import numpy as np
import os
class util:
    def draw(log):
        results={}
        with open(log,"r") as f:
            lines=f.readlines()
            for line in lines:
                key, value = line.split(',')
                results[key]=value
                
        result_dict = {"acc": 0.99, "pre": 0.95, "recall": 0.98, "f1": 0.97}

        keys = np.array(list(result_dict.keys()))
        values = np.array(list(result_dict.values()))
        
        bar_width = 0.5
        
        a = 12
        b = 22
        c  =6
        d = 18
        colors=['#4D1E17','#B43A24','#D89F8A','#F6E1C6']


        plt.bar(keys,values,width=bar_width,color=colors)
        plt.show()
        
    def getRootpath():
        import sys,os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        top_level_path=script_dir
        while True:
            top_level_path, tail = os.path.split(top_level_path)
            if tail == "Evaluation":
                top_level_path=os.path.join(top_level_path,tail)
                break
        sys.path.append(top_level_path)
        from demo1.setRootpath import getRootpath
        ROOT_PATH=getRootpath()
        return ROOT_PATH