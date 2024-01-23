# Get static feature of the program

## usage

```
 python GetStatic.py [project_dir_path]
```

static feature will be stored in ```./static``` 

Intermediate files generated during the process of extracting static features：

```
./
|-- cfg   //raw cfg information extracted from joern-cli-new
|-- cfg_result  //cfg feature 
|-- cut  //slice the code in function level
|   `-- good
|-- out   //Extract all c files in this dir
|-- static  //final result
......
```

