import os
import pandas as pd
import numpy as np
import PIL.Image
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
import sys
from keras.optimizers import SGD
from matplotlib import pylab as plt
from visual_backprop import VisualBackprop
from keras.models import load_model


def show_image(image, op, grayscale=False, ax=None, title=''):
    if ax is None:
        plt.figure()
    plt.axis('off')

    if len(image.shape) == 2 or grayscale is False:
        if len(image.shape) == 3:
            image = np.sum(np.abs(image), axis=2)

        vmax = np.percentile(image, 99)
        vmin = np.min(image)
        plt.imshow(image, vmin=vmin, vmax=vmax)
        plt.title(title)

    else:
        image = image + 127.5
        image = image.astype('uint8')

        plt.imshow(image)
        plt.title(title)
    plt.savefig(op)


def load_image(file_path):
    im = PIL.Image.open(file_path)
    im = np.asarray(im)

    return im - 127.5

RIJKS_MODEL_PATH = sys.argv[1]
RIJKS_WEIGHTS_PATH = sys.argv[2]
csv = sys.argv[3]
res = sys.argv[4]

if not os.path.isdir(res):
    os.makedirs(res)
model = VGG19(weights='imagenet')
model.compile(loss='mean_squared_error', optimizer='adam')

df = pd.read_csv(csv)

for idx, row in df.iterrows():
    img = image.load_img(row['img_path'], target_size=(224, 224))
    img = np.asarray(img)

    # show_image(img, grayscale=False)

    x = np.expand_dims(img, axis=0)

    preds = model.predict(x)
    label = np.argmax(preds)

    visual_bprop = VisualBackprop(model)

    mask = visual_bprop.get_mask(x[0])
    # show_image(mask, ax=plt.subplot('121'), title='ImageNet VisualBackProp')

    trained_model = load_model(RIJKS_MODEL_PATH)
    trained_model.load_weights(RIJKS_WEIGHTS_PATH)

    trained_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss="categorical_crossentropy")

    visual_bprop = VisualBackprop(trained_model)

    mask = visual_bprop.get_mask(x[0])
    op = os.path.join(res, os.path.basename(row['img_path']).split('.')[0] + '_' + 'saliency_map.jpg')
    show_image(mask, op, ax=plt.subplot('121'), title=os.path.basename(row['img_path']) + '_' + row['labels'])
