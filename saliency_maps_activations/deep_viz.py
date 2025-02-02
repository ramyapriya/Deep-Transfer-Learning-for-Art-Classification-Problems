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


def show_image(image, ori_img, op, grayscale=False, title=''):
    fig = plt.figure()
    ax1 = fig.add_subplot('121')
    ax2 = fig.add_subplot('122')
    plt.axis('off')

    if len(image.shape) == 2 or grayscale is False:
        if len(image.shape) == 3:
            image = np.sum(np.abs(image), axis=2)

        vmax = np.percentile(image, 99)
        vmin = np.min(image)
        ax1.imshow(ori_img)
        ax2.imshow(image, vmin=vmin, vmax=vmax)
        plt.suptitle(title)

    else:
        image = image + 127.5
        image = image.astype('uint8')
        ax1.imshow(ori_img)
        ax2.imshow(image)
        plt.suptitle(title)
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

df = pd.read_csv(csv)

model = load_model(RIJKS_MODEL_PATH)
model.load_weights(RIJKS_WEIGHTS_PATH)

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss="categorical_crossentropy")

visual_bprop = VisualBackprop(model)
for idx, row in df.iterrows():
    img = image.load_img(row['img_path'], target_size=(224, 224))
    img = np.asarray(img)
    ori_img = img

    # show_image(img, grayscale=False)

    x = np.expand_dims(img, axis=0)

    preds = model.predict(x)
    label = np.argmax(preds)

    visual_bprop = VisualBackprop(model)

    # show_image(mask, ax=plt.subplot('121'), title='ImageNet VisualBackProp')
    mask = visual_bprop.get_mask(x[0])
    op = os.path.join(res, os.path.basename(row['img_path']).split('.')[0] + '_' + 'saliency_map.jpg')
    show_image(mask, ori_img, op, title=os.path.basename(row['img_path']) + '_' + row['labels'])
