from keras.models import Sequential
from keras.layers import Dense, Flatten


def model_fn(learning_rate):
    """Create a Keras Sequential model with layers."""
    model = Sequential()
    model.add(Flatten(input_shape=(80, 320, 3)))
    model.add(Dense(1))
    compile_model(model, learning_rate)
    return model


def compile_model(model, learning_rate):
    model.compile(loss='mse',
                  optimizer='adam')
    return model
