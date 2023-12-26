#/bin/bash
#choose a smallest file to out put the file content
#params:directory

projectPath="$1"
SCRIPT_ABS_PATH=$(readlink -f "$0")
ROOT_PATH=$(dirname $(dirname $(dirname "$SCRIPT_ABS_PATH")))

dir=$ROOT_PATH"/feature/static/github/static"
if [ -z "$(ls -A $dir)" ]; then
  echo "目录为空"
else
  smallest_file=$(find $dir -type f -exec du {} + | sort -n | head -n 1 | awk '{print $2}')
  #echo $(basename "$smallest_file")
  echo "showing the static code information of "$smallest_file
  cat $smallest_file
fi