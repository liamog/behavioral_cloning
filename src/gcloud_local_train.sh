#!/bin/bash
pushd .
cd "$(dirname "$0")"

X_TRAIN_FILE='../data/proccessed_and_pickled/x_train.pickle'
Y_TRAIN_FILE='../data/proccessed_and_pickled/y_train.pickle'
JOB_DIR=../output/
TRAIN_STEPS=1000
python 'cloud-ml/trainer/task.py' --x_train_file $X_TRAIN_FILE \
                       --y_train_file $Y_TRAIN_FILE \
                       --job-dir $JOB_DIR \
                       --num-epochs 7 \
                       --h5py
popd
