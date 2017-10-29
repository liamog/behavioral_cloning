import argparse
import glob
import json
import os
import numpy as np

import keras
from keras.models import load_model
import model
from tensorflow.python.lib.io import file_io


FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'
BH_CLONE_MODEL = 'bh_clone.hdf5'


def dispatch(x_train_file,
             y_train_file,
             job_dir,
             train_steps,
             train_batch_size,
             learning_rate,
             num_epochs,
             ):
    bh_clone_model = model.model_fn(learning_rate)

    try:
        os.makedirs(job_dir)
    except:
        pass
    print (x_train_file)
    X_train = np.load(x_train_file)
    y_train = np.load(y_train_file)

    bh_clone_model.fit(X_train,
                       y_train,
                       validation_split=0.2,
                       shuffle=True,
                       epochs=num_epochs)
    print(job_dir)
    # Unhappy hack to work around h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over
    # to GCS.
    if job_dir.startswith("gs://"):
        bh_clone_model.save(BH_CLONE_MODEL)
        copy_file_to_gcs(job_dir, BH_CLONE_MODEL)
    else:
        bh_clone_model.save(os.path.join(job_dir, BH_CLONE_MODEL))

# h5py workaround: copy local models over to GCS if the job_dir is GCS.


def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='r') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path),
                            mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_train_file',
                        required=True,
                        type=str,
                        help='Image Training file local or GCS')
    parser.add_argument('--y_train_file',
                        required=True,
                        type=str,
                        help='Steering angle lable file local or GCS')
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints "\
                        "and export model')
    parser.add_argument('--train-steps',
                        type=int,
                        default=100,
                        help="""\
                       Maximum number of training steps to perform
                       Training steps are in the units of training-batch-size.
                       So if train-steps is 500 and train-batch-size if
                       100 then at most 500 * 100 training instances will be
                        used to train.
                      """)
    parser.add_argument('--train-batch-size',
                        type=int,
                        default=40,
                        help='Batch size for training steps')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.003,
                        help='Learning rate for SGD')
    parser.add_argument('--num-epochs',
                        type=int,
                        default=20,
                        help='Maximum number of epochs on which to train')
    parse_args, unknown = parser.parse_known_args()

    dispatch(**parse_args.__dict__)
