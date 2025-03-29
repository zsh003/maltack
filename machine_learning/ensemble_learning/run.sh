#!/bin/bash

start_time=$(date +%s)

python run.py $1
python train_histogram.py
python train_pe_raw.py
python stacking_train.py
python test.py

end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]
echo "Program end. Time cost: $(($cost_time/60))min $(($cost_time%60))s"
