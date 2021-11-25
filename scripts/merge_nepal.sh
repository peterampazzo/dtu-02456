#!/bin/sh

origin="/work1/fbohy/_Data for Frederik and Chris/Nepal video data - Frames and annotation/annotations/"
root="/work3/s203257"

touch ${root}/Nepal_annotation.csv
cd "${origin}"
awk '
  function basename(file) {
    split(file, a, ".")
    return a[1]
  }
  {print $0","basename(FILENAME)}' *.csv > ${root}/Nepal_annotation.csv

