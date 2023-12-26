#/bin/bash
#Extract Program Information (To show dynamic code information 
#params:projectPath souce code'directory

# project_path="$1"
SCRIPT_ABS_PATH=$(readlink -f "$0")
ROOT_PATH=$(dirname $(dirname $(dirname "$SCRIPT_ABS_PATH")))




project_path=$ROOT_PATH"/Evaluation/exampleProject/jasper-version-1.900.1"
dir=$ROOT_PATH"/feature/dynamic/"
shell_file=$dir"main.sh"
dependency_txt=$dir"before_insert.txt"
compile_txt=$dir"compile.txt"
kleecmd_txt=$dir"do.txt"

cd $dir && sh $shell_file $project_path $dependency_txt $compile_txt $kleecmd_txt

