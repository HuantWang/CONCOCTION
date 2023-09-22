#!/bin/bash
for file in $1/*; do
if [ "${file##*.}"x = "ktest"x ];then
        #testcase
        echo "=======testcase========"
        ktest-tool $file
        echo "=========trace========="
        #trace
        klee-replay $2 $file
        echo "======================="
        echo
fi
done

