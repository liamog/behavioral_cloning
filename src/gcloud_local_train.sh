#!/bin/bash
pushd .
cd "$(dirname "$0")"

DATA_DIR="../data/proccessed_and_pickled"
JOB_DIR=$DATA_DIR"_$now/output"

FILES=( right_turn focused_center_on_turns lap_with_mouse gentle_swerving lap2_with_mouse )
# FILES=( lap_with_mouse gentle_swerving )

TRAINING_FILES=""
VALIDATION_FILES=""

for var in "${FILES[@]}"
do
  TRAINING_FILES+="$DATA_DIR/${var}_train.npz "
  VALIDATION_FILES+="$DATA_DIR/${var}_valid.npz "
done

echo $TRAINING_FILES
echo $VALIDATION_FILES

JOB_DIR=../output/

python 'cloud-ml/trainer/task.py' --training_files $TRAINING_FILES \
                       --validation_files $VALIDATION_FILES \
                       --job-dir $JOB_DIR \
                       --num-epochs 3 \
                       --h5py
popd
