#!/bin/bash

vctk_root=$(realpath $1)
filelist_path=$(realpath $2)
vctk_downsampled=$vctk_root/wav_downsampled

mkdir -p $vctk_downsampled
rm $filelist_path

for folder in $(ls -1 $vctk_root/wav48)
do
    mkdir -p $vctk_downsampled/$folder
    for file in $(ls $vctk_root/wav48/$folder/*.wav)
    do
        basefile=$(basename $file)
        
        echo "$file|$(cat $vctk_root/txt/$folder/${basefile%*.wav}.txt)" | tee -a $filelist_path
    done
done
