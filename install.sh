#/bin/bash
#使用root权限执行  bash install.sh /home/soft /home/log 8
#修改安装软件源码（llvm,z3..)地址 log地址（绝对地址） make并行job数
#llvm6.0.1 https://github.com/llvm/llvm-project/blob/release/6.x/llvm/docs/GettingStarted.rst
#klee https://klee.github.io/build-llvm13/
make_j=$3
soft_dir=$1
if [ ! -d "$soft_dir" ]; then
    mkdir -p "$soft_dir"
fi
log_dir=$2
if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

install_jdk=(
  "apt-get install -y python3 python3-pip"
  "apt-get install -y python-numpy "
  "apt install -y openjdk-11-jre-headless "
  "apt install -y openjdk-11-jdk-headless "
  "update-alternatives --config java "
)
install_depen=(
    "apt-get install -y build-essential curl libcap-dev git cmake libncurses5-dev python3-minimal python-pip unzip libtcmalloc-minimal4 libgoogle-perftools-dev libsqlite3-dev doxygen"
    "pip3 install tabulate wllvm"
    "apt install -y gcc g++"
    "pip3 install lit"
    "apt-get install -y zlib1g-dev"
    "apt install -y ninja-build"
)
install_z3=(
    "cd $soft_dir"
    "wget https://github.com/Z3Prover/z3/archive/z3-4.8.8.zip"
    "unzip z3-4.8.8.zip"
    "cd z3-z3-4.8.8"
    "mkdir build"
    "cd build"
    "cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ .."
    "make install"
)

install_klee_uclibc=(
    "cd $soft_dir"
    "git clone https://github.com/klee/klee-uclibc.git"
    "cd klee-uclibc"
    "./configure --make-llvm-lib"
    "make -j $make_j"
)
install_klee=(
    "cd $soft_dir"
    "wget https://github.com/klee/klee/archive/v2.1.zip"
"unzip v2.1.zip"
" cd klee-2.1"
" mkdir build"
" cd build"
" cmake -DENABLE_SOLVER_Z3=ON -DENABLE_POSIX_RUNTIME=ON -DENABLE_KLEE_UCLIBC=ON -DKLEE_UCLIBC_PATH=$soft_dir/klee-uclibc -DLLVM_CONFIG_BINARY=$soft_dir/llvm-project/llvm/build/bin/llvm-config -DLLVMCC=$soft_dir/llvm-project/llvm/build/bin/clang -DLLVMCXX=$soft_dir/llvm-project/llvm/build/bin/clang++ -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ .."
"make -j $make_j"
"make install"
)
install_wllvm=(
"apt-get install -y python3-pip"
"pip3 install --upgrade wllvm"
"wllvm-sanity-checker"
)

log_file_jdk=$log_dir"/install_jdk.log"
log_file_llvm=$log_dir"/install_llvm.log"
log_file_klee=$log_dir"/install_klee.log"
execute_command() {
    local command="$1"
    echo "Running: $command..."
    echo -e "\nRunning: $command..." >> "$2" 
    $command >> "$2" 2>&1
    if [ $? -eq 0 ]; then
        echo "$command executed successfully."
    else
        echo "Error: $command failed. Check $2 for details."
    fi
}

make_llvm(){
    cd $soft_dir
    echo "Running: git clone -b release/6.x --depth 1 https://github.com/llvm/llvm-project.git"
    git clone -b release/6.x --depth 1 https://github.com/llvm/llvm-project.git >>$log_file_llvm 2>&1
    cd llvm-project
    mkdir build
    cd build
    echo "Running: cmake -DLLVM_ENABLE_PROJECTS=clang -DCMAKE_BUILD_TYPE=Release -G \"Unix Makefiles\" ../llvm"
    cmake -DLLVM_ENABLE_PROJECTS=clang -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" ../llvm >>$log_file_llvm 2>&1
    echo "Running: make -j $make_j"
    make -j $make_j >>$log_file_llvm 2>&1
    make install
    llvm-config --version
}


apt-get update

for command in "${install_jdk[@]}"; do
    execute_command "$command" "$log_file_jdk"
done

for command in "${install_depen[@]}"; do
    execute_command "$command" "$log_file_llvm"
done

make_llvm

for command in "${install_z3[@]}"; do
    execute_command "$command" "$log_file_klee"
done

for command in "${install_klee_uclibc[@]}"; do
    execute_command "$command" "$log_file_klee"
done

for command in "${install_klee[@]}"; do
    execute_command "$command" "$log_file_klee"
done

for command in "${install_wllvm[@]}"; do
    execute_command "$command" "$log_file_klee"
done

conda env create -f concoction_environment.yml