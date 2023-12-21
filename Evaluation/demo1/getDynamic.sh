#/bin/bash
SCRIPT_ABS_PATH=$(readlink -f "$0")
ROOT_PATH=$(dirname $(dirname $(dirname "$SCRIPT_ABS_PATH")))

func(){
    ROOT_PATH=$(dirname $(dirname $(dirname "$SCRIPT_ABS_PATH")))
    dirr=$ROOT_PATH"/concoction/data/dataset0_dy"
    if [ -z "$(ls -A $dirr)" ]; then
        echo "目录为空"
    else
        smallest_file=$(find $dirr -type f -exec du {} + | sort -n | head -n 1 | awk '{print $2}')
        #echo $(basename "$smallest_file")
        echo "showing the dynamic code information of "$smallest_file
        cat $smallest_file
    fi
}

# 指令和超时时间
command_to_execute="sh /homee/Evaluation/demo1/execDynamic.sh"
timeout_duration=60
timeout 60s sh /homee/Evaluation/demo1/execDynamic.sh
if [ $? -eq 124 ]; then
    echo "timeout"
fi
func