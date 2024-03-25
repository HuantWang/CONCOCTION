# Get hybid feature of the program

**Environment requirements [klee docker]: features are processed in the klee tool's container** 

```
KLEE 2.1 (https://klee.github.io)
  Build mode: RelWithDebInfo (Asserts: ON)
  Build revision: unknown

LLVM (http://llvm.org/):
  LLVM version 6.0.1
  Optimized build.
  Default target: x86_64-unknown-linux-gnu
  Host CPU: skylake-avx512
  
clang version ：6.0.1
```

**Pull the official klee image directly from Linux and create a docker container.**

```
$ docker pull klee/klee:2.1
$ docker run -dit --name=dynamic_klee --ulimit='stack=-1:-1' klee/klee:2.1
$ docker exec -it -p hostPort:dockerPort dynamic_klee2.1 /bin/bash
```

**Normally, the relevant tools are ready in docker.**

| Code Catalog                                              | functionality                                             |
| --------------------------------------------------------- | --------------------------------------------------------- |
| /home/feature/dynamic/0before_insert.py                   | Configure the environment required to compile the project |
| /home/feature/dynamic/1insert.py                          | Staking the code                                          |
| /home/feature/dynamic/2compile.py                         | Compile the project                                       |
| /home/feature/dynamic/3get_bc.py                          | Acquisition of dynamic features                           |
| /home/feature/dynamic/python-do/3.1replay.sh              | Recurrent paths as dynamic features                       |
| /home/feature/dynamic/4get_composite.py                   | Getting hybrid features                                   |
| /home/feature/dynamic/python-do/4.1cut.py                 | Slicing dynamic features by input                         |
| /home/feature/dynamic/python-do/4.2automated.py           | Getting hybrid features                                   |
| /home/feature/dynamic/python-do/4.2.1extract_file-func.py | Slicing dynamic features by function                      |
| /home/feature/dynamic/python-do/4.2.2insert_testcase.py   | Input for the feature complementary function              |
| /home/feature/dynamic/python-do/4.2.3before_concat.py     | Complete preparations for hybrid feature acquisition      |
| /home/feature/dynamic/python-do/4.2.4concat.py            | Getting hybrid features                                   |
| /home/feature/dynamic/python-do/4.2.5after_concat.py      | Deletion of incomplete features                           |
| /home/feature/dynamic/python-do/4.2.6delet_dynamicfile.py | Deletion of incomplete features                           |

Notes: 

3get_bc.py calls 3.1 replay.sh

4get_composite.py code calls all scripts from 4.1 to 4.2.6

# usage

1、Staking the code

```
$ python3 before_insert.py home/feature/dynamic/0before_insert.txt
# The file 0before_insert.txt contains instructions for installing all the libraries. The script installs the libraries that the project depends on by running the instructions in the file.

$ python3 1insert.py /home/feature/jasper-version-1.900.1 
# The jasper-version-1.900.1 file is the source code address of the project to be staked, and the output is the source code of the project after staking.
```

2、Compiling

```
$ python3 2compile.py /home/feature/jasper-version-1.900.1  /home/feature/dynamic/compile.txt
# The compile.txt file contains a txt file of wllvm compilation instructions. The script compiles the project by running the instructions in the file to obtain a bitcode file.
```

3、Hybrid Feature Acquisition

```
$ python3 get_composite.py 
# Get the composite folder where the hybrid features are stored, in the /home/feature/dynamic/ directory
```





# extract dynamic feature
## dependency
install klee and llvm
```
# klee -version
KLEE 2.1 (https://klee.github.io)
  Build mode: RelWithDebInfo (Asserts: ON)
  Build revision: unknown

LLVM (http://llvm.org/):
  LLVM version 6.0.1
  Optimized build.
  Default target: x86_64-unknown-linux-gnu
  Host CPU: skylake-avx512
```
## run the example
extract the dynamic feature from project 'jasper1.9.1'
```
sh main.sh /home/example/jasper-version-1.900.1 ./before_insert.txt ./compile.txt ./do.txt
```