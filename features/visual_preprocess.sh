#!/usr/bin/env bash


python photos_dump.py
echo "Photo dumped......"
cores=`cat /proc/cpuinfo |grep "processor"|wc -l`

for ((i=0; i<$cores; i++));
do
    (python visual_zip_process.py -k $i) &
done
echo "Wating all sub matrix extracting......"
wait
echo "All sub matrix extracted......"
echo "Merging all sub matrix"
python visual_matrix_merge.py