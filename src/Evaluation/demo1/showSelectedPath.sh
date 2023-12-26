#/bin/bash
#choose a smallest file to out put the file content
#params:directory

projectPath="$1"
SCRIPT_ABS_PATH=$(readlink -f "$0")
ROOT_PATH=$(dirname $(dirname $(dirname "$SCRIPT_ABS_PATH")))

dir2=$ROOT_PATH"/concoction/data/dataset0"
dir=$ROOT_PATH"/concoction/data/feature_path_text"
# if [ -z "$(ls -A $dir)" ]; then
#   echo "目录为空"
# else
#   smallest_file=$(find $dir -type f -exec du {} + | sort -n | head -n 1 | awk '{print $2}')
# fi

files=("$dir"/*)

# 检查目录是否为空
if [ ${#files[@]} -eq 0 ]; then
  echo "目录为空"
  exit 1
fi

func(){
  # 随机选择一个文件
  selected_file="${files[RANDOM % ${#files[@]}]}"
  echo $selected_file
  smallest_file=$selected_file

  found_files=$(find "$dir2" -type f -name "$(basename "$smallest_file")")

  if [ -n "$found_files" ]; then
      echo "$found_files"
  else
      echo "文件不存在."
  fi


  # echo $smallest_file 
  echo $(basename "$smallest_file")"\' path nums: "
  python $(dirname "$SCRIPT_ABS_PATH")"/getPathNum.py" $found_files

  echo "selected path:"

  echo $(basename "$smallest_file")
  echo "showing the code and the selected path of "$smallest_file
  cat $smallest_file
}

file=$ROOT_PATH"/concoction/data/feature_path_text/jas_cm.c-jas_cmprof_createfromiccprof.c.txt"
if [ -e "$file" ]; then

    found_files=$(find "$dir2" -type f -name "$(basename "$file")")
    if [ -n "$found_files" ]; then
        echo "$found_files"
    else
        echo "file not found."
    fi


    # echo $smallest_file 
    echo $(basename "$smallest_file")" path nums: "
    python $(dirname "$SCRIPT_ABS_PATH")"/getPathNum.py" $found_files
    echo "showing the code and the selected path of "$file
    cat "$file"
else
    func
fi
