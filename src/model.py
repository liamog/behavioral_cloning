from keras.models import load_model


model = load_model('/Users/liamog/udacity/behavioural_cloning/src/output/weights-improvement-00-0.17.hdf5')
print(model.summary())
