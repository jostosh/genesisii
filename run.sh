#!/bin/bash

declare -a optimizer=("eve" "adam" "genesis")
declare -a data=("oxflower" "mnist" "cifar100","cifar10")
## now loop through the above array
for opt in "${optimizer[@]}"
    do
    for dat in "${data[@]}"
    do
        python3 train.py --optimizer $opt --data $dat
       # or do whatever with individual element of the array
    done
done
