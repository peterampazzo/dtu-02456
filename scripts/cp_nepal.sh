#!/bin/sh

root="/work3/s203257/"
origin="/work1/fbohy/_Data for Frederik and Chris/Nepal video data - Frames and annotation/sample_frames/"
folder="Nepal_raw"

cd ${root}
mkdir ${folder}
cp -R "$origin"* ${root}${folder}/

cd ${root}${folder}/
echo "This files are going to be deleted:"
find . -name "*.ini" -type f

find . -name "*.ini" -type f -delete
