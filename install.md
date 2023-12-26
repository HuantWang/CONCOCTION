# Docker
Our Docker images are one of the fastest ways to get started.
# Manual Installation
## 1.jdk11
```
$ apt-get update
$ apt-get install vim
$ apt-get install python3 python3-pip
$ apt-get install python-numpy
$ apt install openjdk-11-jre-headless
$ apt install openjdk-11-jdk-headless
$ sudo update-alternatives --config java
```
## 2.KLEE 2.1

2.1 Install dependencies
```
$ apt-get install build-essential curl libcap-dev git cmake libncurses5-dev
python3-minimal python-pip unzip libtcmalloc-minimal4 libgoogle-perftools-dev
libsqlite3-dev doxygen
$ pip3 install tabulate wllvm
$ apt install gcc g++
$ pip3 install lit
$ apt-get install zlib1g-dev
```
2.2 LLVM 6.0.1
```
$ cd llvm-project-llvmorg-6.0.1
$ mkdir build
$ cd build
$ cmake -DLLVM_ENABLE_PROJECTS=clang -DCMAKE_BUILD_TYPE=Release -G "Unix
Makefiles" ../llvm
$ make -j 4
$ make install
$ llvm-config --version
```
2.3 z3
```
$ wget https://github.com/Z3Prover/z3/archive/z3-4.8.8.zip
$ unzip z3-4.8.8.zip
$ cd z3-z3-4.8.8
$ mkdir build
$ cd build
$ cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
$ make install
```
2.4 klee-uclibc
```
$ git clone https://github.com/klee/klee-uclibc.git
$ cd klee-uclibc
$ ./configure --make-llvm-lib
$ make -j 8
```
2.5 WLLVM
```
$ apt-get install update
$ apt-get install python3-pip
$ pip3 install --upgrade wllvm
$ wllvm-sanity-checker 
```
2.6 KLEE
```
$ wget https://github.com/klee/klee/archive/v2.1.zip
$ unzip v2.1.zip
$ cd klee-2.1
$ mkdir build
$ cd build
$ cmake -DENABLE_SOLVER_Z3=ON -DENABLE_POSIX_RUNTIME=ON -DENABLE_KLEE_UCLIBC=ON -
DKLEE_UCLIBC_PATH=<KLEE_UCLIBC_SOURCE_DIR> -
DLLVM_CONFIG_BINARY<LLVM_DIR/llvm-config> -
DLLVMCC=<PATH_TO_CLANG> -DLLVMCXX=<PATH_TO_CLANG++> -DCMAKE_C_COMPILER=clang -
DCMAKE_CXX_COMPILER=clang++ ..
$ make -j8
$ make install
```
## 3.python
```
conda env create -f concoction_environment.yml
```