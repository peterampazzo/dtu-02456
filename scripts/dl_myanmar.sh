#!/bin/sh

# Download Myanmar dataset from https://osf.io/4pwj8/
# The data set contains 7 zip files (HELMET_DATASET > image)
# The script downloads all of them, unzip and merge all the subset in a single folder

root="/work3/s203247/"
folder="Myanmar_raw"
count=1
parts=(
    "https://osf.io/452nq/download"
    "https://osf.io/muzgb/download"
    "https://osf.io/9dmw3/download"
    "https://osf.io/39ynb/download"
    "https://osf.io/jyg9s/download"
    "https://osf.io/6h5ka/download"
    )

# cd ${root}
mkdir ${folder}
cd ${folder}

for set in "${parts[@]}"
do
    echo ${set}
    wget ${set}
    unzip download
    rm download
    cd part_${count}/
    mv * ../
    cd ..
    rm -rf part_${count}/
    ((count=count+1))
done

# Download csv file with data split
cd ..
wget https://osf.io/q7rmb/download
mv download Myanmar_data_split.csv

# Download csv file with annotation
wget https://osf.io/buh57/download
unzip download
rm download
mv annotation Myanmar_annotations
