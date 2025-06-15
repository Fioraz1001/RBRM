#!/bin/bash
sg='Safe_guard.py'
mj='salad.py'

doo() {
    local rp=$1
    local wp=$2
    local func=$3
    python $func --data_path=$rp --write_path=$wp
}


MODEL_NAME=YOUR_MODEL_PATH_HERE
WRITE_PATH=YOUR_SAMPLE_PATH_HERE

python Salad-bench.py --data_path=./base_set.jsonl --model_path=$MODEL_NAME --write_path=$WRITE_PATH --length=4000

doo $WRITE_PATH $WRITE_PATH $sg
doo $WRITE_PATH $WRITE_PATH $mj








