"""Udacity Self Driving Car nano degree - behavioral cloning project."""
import argparse
import os
import numpy as np
from os import walk
from os import makedirs
from io import BytesIO

from tensorflow.python.lib.io import file_io

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

from sklearn.utils import shuffle

BH_CLONE_MODEL = 'bh_clone.hdf5'

input_shape = (80, 320, 3)


def model_fn():
    """Create a Keras Sequential model with layers."""
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Dropout(0.25))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(1))
    compile_model(model)
    return model


def compile_model(model):
    """
    Compiles Keras model
    """
    opt = Adam(lr=.001, beta_1=0.9,
               beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mse',
                  optimizer=opt, metrics=['accuracy'])
    return model


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
    return (int(num_samples), int(num_samples / batch_size))


def generator(files, batch_size=32):
    while 1:  # Loop forever so the generator never terminates
        shuffle(files)
        for file in files:
            print("Serving from " + file)
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
             num_epochs,
             ):
    bh_clone_model = model_fn()
    local_job_dir = "output"
    makedirs(local_job_dir, exist_ok=True)
    training_len, training_steps = data_sizes(training_files)
    validation_len, validation_steps = data_sizes(validation_files)
    print('training_size = {} , training_steps={}'.format(
        training_len, training_steps))
    print('validation_size={}, validation_steps={}'.format(
        validation_len, validation_steps))
    # compile and train the model using the generator function
    train_generator = generator(training_files, batch_size=32)
    validation_generator = generator(validation_files, batch_size=32)
    makedirs(job_dir, exist_ok=True)
    print(job_dir)

    chceckpoint_file = local_job_dir + \
        "/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
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
                                 samples_per_epoch=training_steps,
                                 nb_val_samples=validation_steps,
                                 nb_epoch=num_epochs)
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
    parser.add_argument('--num-epochs',
                        type=int,
                        default=20,
                        help='Maximum number of epochs on which to train')
    parse_args, unknown = parser.parse_known_args()

    dispatch(**parse_args.__dict__)
