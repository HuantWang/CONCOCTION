startTime=`date +"%Y-%m-%d %H:%M:%S"`
python /home/vuldeepker/vuldeepecker.py
endTime=`date +"%Y-%m-%d %H:%M:%S"`
st=`date -d  "$startTime" +%s`
et=`date -d  "$endTime" +%s`
sumTime=$(($et-$st))
echo "Total time is : $sumTime second."
