#/bin/bash
TRAIN_FILE=adult.data.csv
EVAL_FILE=adult.test.csv

GCS_TRAIN_FILE=gs://liamog_udacity/bh_cloning/data/x_train.pickle
GCS_EVAL_FILE=gs://liamog_udacity/bh_cloning/data/y_train.pickle

gsutil cp $GCS_TRAIN_FILE $TRAIN_FILE
gsutil cp $GCS_EVAL_FILE $EVAL_FILE

JOB_DIR=bh_cloning_keras
TRAIN_STEPS=1000
python trainer/task.py --train-files $TRAIN_FILE \
                       --eval-files $EVAL_FILE \
                       --job-dir $JOB_DIR \
                       --train-steps $TRAIN_STEPS
