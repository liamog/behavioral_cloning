import argparse
import os
import numpy as np
from os import walk

from io import BytesIO
import model
from tensorflow.python.lib.io import file_io

from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'
BH_CLONE_MODEL = 'bh_clone.hdf5'


def load_data(file):
    if file.startswith("gs://"):
        f = BytesIO(file_io.read_file_to_string(file, binary_mode=True))
        return np.load(f)
    else:
        return np.load(file)


def data_sizes(files, batch_size=32):
    num_samples = 0
    for file in files:
        uncompressed = load_data(file)
        X_samples = uncompressed['x']
        num_samples += len(X_samples)
    return (num_samples, num_samples / batch_size)


def generator(files, batch_size=32):
    while 1:  # Loop forever so the generator never terminates
        shuffle(files)
        for file in files:
            print()
            print("Serving from " + file)
            print()
            uncompressed = load_data(file)
            X_samples = uncompressed['x']
            y_samples = uncompressed['y']
            shuffle(X_samples, y_samples)
            num_samples = len(X_samples)
            for offset in range(0, num_samples, batch_size):
                X_batch_samples = X_samples[offset:offset + batch_size]
                y_batch_samples = y_samples[offset:offset + batch_size]
                yield (X_batch_samples, y_batch_samples)


def dispatch(training_files,
             validation_files,
             job_dir,
             learning_rate,
             num_epochs,
             ):
    bh_clone_model = model.model_fn(learning_rate)
    local_job_dir = "local_output"
    try:
        os.makedirs(local_job_dir)
    except:
        pass
    training_len, training_steps = data_sizes(training_files)
    validation_len, validation_steps = data_sizes(validation_files)
    print('training_size = {} , training_steps={}'.format(
        training_len, training_steps))
    print('validation_size={}, validation_steps={}'.format(
        validation_len, validation_steps))

    # compile and train the model using the generator function
    train_generator = generator(training_files, batch_size=32)
    validation_generator = generator(validation_files, batch_size=32)

    chceckpoint_file = local_job_dir + \
        "/weights-improvement--{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(
        chceckpoint_file,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max')
    callbacks_list = [checkpoint]
    # Fit the model
    bh_clone_model.fit_generator(generator=train_generator,
                                 validation_data=validation_generator,
                                 verbose=1,
                                 callbacks=callbacks_list,
                                 steps_per_epoch=training_steps,
                                 validation_steps=validation_steps,
                                 epochs=num_epochs)
    bh_clone_model.save(os.path.join(local_job_dir, BH_CLONE_MODEL))

    # Get list of all output files
    output_files = []
    for (dirpath, dirnames, filenames) in walk(local_job_dir):
        output_files.extend(filenames)
        break

    for filename in output_files:
        copy_file_to_jobdir(job_dir, local_job_dir, filename)


def copy_file_to_jobdir(job_dir, local_job_dir, filename):
    src_file = os.path.join(local_job_dir, filename)
    dst_file = os.path.join(job_dir, filename)
    with file_io.FileIO(src_file, mode='r') as input_f:
        with file_io.FileIO(dst_file,
                            mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_files',
                        required=True,
                        type=str,
                        help='Training file local or GCS', nargs='+')
    parser.add_argument('--validation_files',
                        required=True,
                        type=str,
                        help='Validation file local or GCS', nargs='+')
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints "\
                        "and export model')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.003,
                        help='Learning rate for adam')
    parser.add_argument('--num-epochs',
                        type=int,
                        default=20,
                        help='Maximum number of epochs on which to train')
    parse_args, unknown = parser.parse_known_args()

    dispatch(**parse_args.__dict__)
