#!/bin/bash
pushd .
cd "$(dirname "$0")"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="bh_clone_$now"
GS_DIR="gs://liamog_udacity/bh_cloning"

JOB_DIR=$GS_DIR"_$now/output"

TRAIN_FILE=$GS_DIR"/train.npz"
EPOCHS=2

echo $X_TRAIN_FILE

gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --runtime-version 1.2 \
                                    --job-dir $JOB_DIR \
                                    --package-path cloud-ml/trainer \
                                    --module-name trainer.task \
                                    --region us-central1 \
                                    -- \
                                    --train_file $TRAIN_FILE \
                                    --job-dir $JOB_DIR \
                                    --num-epochs $EPOCHS
popd
