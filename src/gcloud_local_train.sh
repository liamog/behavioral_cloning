#!/bin/bash
pushd .
cd "$(dirname "$0")"

TRAINING_FILES='../data/proccessed_and_pickled/swerving_full_train.npz ../data/proccessed_and_pickled/lap_with_mouse_train.npz ../data/proccessed_and_pickled/track1_train.npz'
VALIDAION_FILES='../data/proccessed_and_pickled/swerving_full_valid.npz ../data/proccessed_and_pickled/lap_with_mouse_valid.npz ../data/proccessed_and_pickled/track1_valid.npz'
JOB_DIR=../output/

python 'cloud-ml/trainer/task.py' --training_files $TRAINING_FILES \
                       --validation_files $VALIDAION_FILES \
                       --job-dir $JOB_DIR \
                       --num-epochs 5 \
                       --h5py
popd
