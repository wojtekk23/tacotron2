#!/bin/bash

vctk_root=$(realpath $1)
filelist_path=$(realpath $2)
vctk_downsampled=$vctk_root/wav_downsampled

mkdir -p $vctk_downsampled
rm -f $filelist_path
touch $filelist_path

for folder in $(ls -1 $vctk_root/wav48 | head -n -11)
do
    mkdir -p $vctk_downsampled/$folder
    for file in $(ls $vctk_root/wav48/$folder/*.wav)
    do
        basefile=$(basename $file)
        
        echo "$file|$(cat $vctk_root/txt/$folder/${basefile%*.wav}.txt | sed -e 's/^"//' -e 's/"$//')" | tee -a $filelist_path
    done
done

mv $filelist_path $filelist_path.tmp
cat $filelist_path.tmp | shuf > $filelist_path
rm -f $filelist_path.tmp

# Train/validation split (80:20 not including the test data)
head -n -8000 $filelist_path > ${filelist_path%*.txt}_train.txt
tail -n  8000 $filelist_path > ${filelist_path%*.txt}_valid.txt
