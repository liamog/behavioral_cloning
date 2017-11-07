#!/bin/bash
pushd .
cd "$(dirname "$0")"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="bh_clone_$now"
DATA_DIR="gs://liamog_udacity/bh_cloning"

JOB_DIR=$DATA_DIR"_$now/output"
FILES=( right_turn cc_lap_with_mouse lap_with_mouse lap2_with_mouse gentle_swerving track1 track2)

TRAINING_FILES=""
VALIDATION_FILES=""

for var in "${FILES[@]}"
do
  TRAINING_FILES+="$DATA_DIR/${var}_train.npz "
  VALIDATION_FILES+="$DATA_DIR/${var}_valid.npz "
  # do something on $var
done

echo $TRAINING_FILES
echo $VALIDATION_FILES
#From testing, looks like we get our best validation result after 2 epcohs.
EPOCHS=2

gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --runtime-version 1.2 \
                                    --job-dir $JOB_DIR \
                                    --package-path cloud-ml/trainer \
                                    --module-name trainer.task \
                                    --region us-central1 \
                                    -- \
                                    --training_files $TRAINING_FILES \
                                    --validation_files $VALIDATION_FILES \
                                    --job-dir $JOB_DIR \
                                    --num-epochs $EPOCHS
popd
