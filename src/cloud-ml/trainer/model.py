from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
input_shape = (80, 320, 3)


def model_fn(learning_rate):
    """Create a Keras Sequential model with layers."""
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Dropout(0.25))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(1))
    compile_model(model, learning_rate)
    return model


def compile_model(model, learning_rate):
    opt = Adam(lr=learning_rate, beta_1=0.9,
               beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mse',
                  optimizer=opt, metrics=['accuracy'])
    return model
