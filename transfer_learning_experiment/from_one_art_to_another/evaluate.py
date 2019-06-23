from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import keras
import pandas as pd
from keras.utils import np_utils
from keras.models import Model, load_model
from keras.layers import Dense
from collections import Counter
import numpy as np
import os
import h5py

import sys


def my_generator(self, mode, X_test, y_test, size=224, channels=3):

    if mode == "__test":
        X_ = X_test
        y_ = y_test
        batch_size = len(X_)

    start_batch = 0
    end_batch = start_batch + batch_size
    while True:

        # Returns a random batch indefinitely from X_train, needed also in order to catch exception

        batch = list()

        if len(X_) - end_batch < 0:
            start_batch = start_batch - batch_size + 1
            end_batch = end_batch - batch_size + 1

        for imgs in X_[start_batch:end_batch]:
            img = image.load_img(io.BytesIO(imgs), target_size=(size, size))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            batch.append(img)

        batch = np.asarray(batch)

        X_batch = np.reshape(
            batch, (batch.shape[0], size, size, channels))
        y_batch = np.asarray([item for item in y_[start_batch:end_batch]])

        yield(X_batch, y_batch)


def one_hot_encoding(total_labels):
    one_hot_encodings = list()
    encoder = LabelEncoder()
    n_labels = len(Counter(total_labels).keys())
    encoded_y = encoder.fit_transform(total_labels)
    one_hot_encodings = np_utils.to_categorical(encoded_y, n_labels)
    print (one_hot_encodings)
    return one_hot_encodings, n_labels


def store_images_to_hdf5(path, images, split='test'):
    f = h5py.File(path)
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    dset = f.create_dataset(split, (len(images), ), dtype=dt)
    for i in range(0, len(images)):

        filename = images[i]
        fin = open(filename, 'rb')
        binary_data = fin.read()

        dset[i] = np.fromstring(binary_data, dtype='uint8')

def load_images(path, split):

    f = h5py.File(path, 'r')
    images = list(f[split])

    return(images)

def load_encodings(path, split):
    h5f_labels = h5py.File(path, 'r')
    labels = h5f_labels[split][:]

    return(labels)


def store_encodings_to_hdf5(path, encodings, split='test'):
    f = h5py.File(path)
    dset = f.create_dataset(split, data=encodings)


csv = sys.argv[1]
hdf5_path = sys.argv[2]
model_path = sys.argv[3]
df = pd.read_csv(csv)
images = df['img_path'].tolist()
labels = df['labels'].tolist()

one_hot_encodings, n_labels = one_hot_encoding(labels)

testing_images_path = os.path.join(hdf5_path, "testing_images.hdf5")
testing_labels_path = os.path.join(hdf5_path, "testing_labels.hdf5")

if not os.path.exists(testing_images_path) or not os.path.exists(testing_labels_path):
    store_images_to_hdf5(testing_images_path, images, 'X_test')
    store_encodings_to_hdf5(testing_labels_path, one_hot_encodings, 'y_test')
else:
    print ('hdf5 paths already exist!')

X_test = load_images(testing_images_path, 'X_test')
y_test = load_encodings(testing_labels_path, 'y_test')

model = load_model(model_path)

tl_score = model.evaluate_generator(my_generator('__test', X_test, y_test), len(X_test))
print('Test accuracy via Transfer-Learning:', tl_score[1])
