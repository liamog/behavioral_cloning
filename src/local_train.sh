#!/bin/bash
pushd .
cd "$(dirname "$0")"

DATA_DIR="../data/proccessed_and_pickled"
JOB_DIR="output_$now/"

#FILES=( right_turn focused_center_on_turns lap_with_mouse gentle_swerving lap2_with_mouse )
FILES=( bridge right_turn gentle_swerving focused_center_on_turns lap_with_mouse lap2_with_mouse cc_lap_with_mouse)

TRAINING_FILES=""
VALIDATION_FILES=""

for var in "${FILES[@]}"
do
  TRAINING_FILES+="$DATA_DIR/${var}_train.npz "
  VALIDATION_FILES+="$DATA_DIR/${var}_valid.npz "
done

echo $TRAINING_FILES
echo $VALIDATION_FILES

python 'standalone/local.py' --training_files $TRAINING_FILES \
                       --validation_files $VALIDATION_FILES \
                       --job-dir $JOB_DIR \
                       --num-epochs 2 \
                       --h5py
popd
