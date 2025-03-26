#!/bin/bash


# Intra training and testing
MAX_ROUND=1
conv="LDC_M"
EPOCH=100
BS=3 

for round in $(seq 1 $MAX_ROUND)
do
    echo Round \#"$round"
    for dataset in P U
    do
        for k in 2 
        do  
            echo python3 train.py --train_dataset=$dataset --test_dataset=$dataset --conv=$conv --model_S $k --bs $BS --epoch $EPOCH
            python3 train.py --train_dataset=$dataset --test_dataset=$dataset --conv=$conv --model_S $k --bs $BS --epoch $EPOCH
            
            echo python3 test.py --train_dataset=$dataset --test_dataset=$dataset --conv=$conv --model_S $k --epoch $EPOCH 
            python3 test.py --train_dataset=$dataset --test_dataset=$dataset --conv=$conv --model_S $k --epoch $EPOCH

        done
    done
done

