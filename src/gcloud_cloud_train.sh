#!/bin/bash
pushd .
cd "$(dirname "$0")"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="bh_clone_$now"
GS_DIR="gs://liamog_udacity/bh_cloning"

JOB_DIR=$GS_DIR"_$now/output"

TRAINING_FILES="$GS_DIR/right_turn_train.npz $GS_DIR/swerving_trimmed_train.npz $GS_DIR/lap_with_mouse_train.npz $GS_DIR/track1_train.npz"
VALIDATION_FILES="$GS_DIR/right_turn_valid.npz $GS_DIR/swerving_trimmed_valid.npz $GS_DIR/lap_with_mouse_valid.npz $GS_DIR/track1_valid.npz"

echo $TRAINING_FILES
echo $VALIDATION_FILES
EPOCHS=5

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
