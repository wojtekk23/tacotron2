#!/bin/bash

vctk_root=$(realpath $1)
filelist_path=$(realpath $2)
vctk_downsampled=$vctk_root/wav_downsampled

mkdir -p $vctk_downsampled
rm -f $filelist_path
touch $filelist_path

for folder in $(ls -1 $vctk_root/wav48 | tail -n 11)
do
    mkdir -p $vctk_downsampled/$folder
    for file in $(ls $vctk_root/wav48/$folder/*.wav)
    do
        basefile=$(basename $file)
        
        echo "$file|$(cat $vctk_root/txt/$folder/${basefile%*.wav}.txt | sed -e 's/^"//' -e 's/"$//')" | tee -a $filelist_path
    done
done
