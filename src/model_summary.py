from keras.models import load_model
from keras.utils.vis_utils import plot_model


model = load_model('/home/liam/udacity/behavioral_cloning/src/output/bh_clone.hdf5')
print(model.summary())

plot_model(model, to_file='model.png', show_shapes=True)