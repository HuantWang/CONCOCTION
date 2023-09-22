#!/bin/bash
#sh main.sh project_path 依赖安装指导文件 编译指导文件 运行指导文件  (给绝对路径)
#sh main.sh /home/jasper-version-1.900.1/jasper-version-1.900.1 /home/feature/dynamic/before_insert.txt /home/feature/dynamic/compile.txt /home/feature/dynamic/do.txt
check_cmd() {
  local succeed_message="$1" 
  local failed_message="$2"  
  if [ $? -eq 0 ]; then
    echo $succeed_message
   else
    echo $failed_message
    exit 1 
  fi
}
SCRIPT_ABS_PATH=$(readlink -f "$0")
SCRIPT_ABS_DIR=$(dirname "$SCRIPT_ABS_PATH")
SCRIPT_ABS_DIR_static=$(dirname "$SCRIPT_ABS_DIR")
echo $SCRIPT_ABS_PATH
echo $SCRIPT_ABS_DIR

if [ -d "$SCRIPT_ABS_DIR_static/static/github/static" ]; then
  echo "static ffeature exisit"
else
  GetStatic_py_path="$SCRIPT_ABS_DIR_static/static/github/GetStatic.py"
  echo "==========================begin to extract static feature============="
  cd $SCRIPT_ABS_DIR_static
  python3 $GetStatic_py_path $1
  check_cmd "==========================succeed to extract static feature==========================" "==========================failed to extract static feature=========================="
fi


cd $SCRIPT_ABS_DIR
echo "==========================begin to prepare the dependency============="
python3 before_insert.py $2
check_cmd "==========================succeed to prepare the dependency==========================" "==========================failed to prepare the dependency=========================="
echo "==========================begin to insert the program============="
python3 insert.py $1
check_cmd "==========================succeed to insert the program==========================" "==========================failed to insert the program=========================="
echo "==========================begin to compile the program============="
python3  compile.py  $1 $3
check_cmd "==========================succeed to compile the program==========================" "==========================failed to compile the program=========================="

echo "==========================begin to klee the program============="
python3  get_bc.py  $1 $4
check_cmd "==========================succeed to use klee==========================" "==========================failed to use klee=========================="

echo "==========================extract dynamic feature and concate============="
python3  get_composite.py 
check_cmd "==========================succeed to concate==========================" "==========================failed to concate=========================="

echo "==========================cancate============="
python3  get_compositeee.py $SCRIPT_ABS_DIR"/dynamic/jasper/result/static" $SCRIPT_ABS_DIR"/dynamic/jasper/result/dynamic" $SCRIPT_ABS_DIR"/dynamic/jasper/result/composite"
check_cmd "==========================get_compositeee.py  endd==========================" "==========================failed to concate=========================="

