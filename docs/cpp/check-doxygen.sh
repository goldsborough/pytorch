#!/bin/bash

set -ex

ignore_warning() {
  # Invert match to filter out $1.
  grep -v "$1" doxygen-log.txt > temp.txt
  mv temp.txt doxygen-log.txt
}

doxygen 2> original-doxygen-log.txt
cp original-doxygen-log.txt doxygen-log.txt

ignore_warning "warning: no uniquely matching class member found for"
ignore_warning "warning: source ../../build/aten/src/ATen/Tensor.h is not a readable file"
ignore_warning "warning: source ../../build/aten/src/ATen/Functions.h is not a readable file"

warnings=$(grep 'warning:' doxygen-log.txt | wc -l)

if [[ $warnings != 0 ]]; then
  cat original-doxygen-log.txt
  exit 1
fi
