# coding: utf-8
from imagenet_vector import get_imagenet_vector
from keras.models import load_model
m = load_model('/media/dev_hdd2/ramya/art_style_exps/approaches/iter2/results/Art_Style_Macys/from_one_art_to_another/VGG19/TL_VGG19_model.h5')
m.load_weights('/media/dev_hdd2/ramya/art_style_exps/approaches/iter2/results/Art_Style_Macys/from_one_art_to_another/VGG19/TL_VGG19_weights.h5')
df = pd.read_csv('/media/train_hdd2/Macys_phase3/art_testset/art_style_testset_inp_valid.csv')
df.iloc[0]
import pandas as pd
df = pd.read_csv('/media/train_hdd2/Macys_phase3/art_testset/art_style_testset_inp_valid.csv')
df.iloc[0]
df.labels.value_counts()
df = pd.read_csv('/media/train_hdd2/Macys_phase3/art_testset/art_style_testset_inp_valid_reduced_7cls_cropped_images.csv')
df.head()
k = []
for idx, row in df.iterrows():
    img = image.load_img(row['img_path'], target_size=(size,size))
    img = image.img_to_array(img)
    img = preprocess_input(np.expand_dims(img, axis=0))
    pred = m.predict(img)
    res = classes[np.argmax(pred)]
    pred = pred[0].tolist()
    df.loc[idx, 'predictions'] = str([round(i,3) for i in pred])
    df.loc[idx, 'pred'] = res
    print (row['img_path'])
    
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
k = []
for idx, row in df.iterrows():
    img = image.load_img(row['img_path'], target_size=(size,size))
    img = image.img_to_array(img)
    img = preprocess_input(np.expand_dims(img, axis=0))
    pred = m.predict(img)
    res = classes[np.argmax(pred)]
    pred = pred[0].tolist()
    df.loc[idx, 'predictions'] = str([round(i,3) for i in pred])
    df.loc[idx, 'pred'] = res
    print (row['img_path'])
    
size = 224
channels = 3
k = []
for idx, row in df.iterrows():
    img = image.load_img(row['img_path'], target_size=(size,size))
    img = image.img_to_array(img)
    img = preprocess_input(np.expand_dims(img, axis=0))
    pred = m.predict(img)
    res = classes[np.argmax(pred)]
    pred = pred[0].tolist()
    df.loc[idx, 'predictions'] = str([round(i,3) for i in pred])
    df.loc[idx, 'pred'] = res
    print (row['img_path'])
    
import pandas as pd
import numpy as np
k = []
for idx, row in df.iterrows():
    img = image.load_img(row['img_path'], target_size=(size,size))
    img = image.img_to_array(img)
    img = preprocess_input(np.expand_dims(img, axis=0))
    pred = m.predict(img)
    res = classes[np.argmax(pred)]
    pred = pred[0].tolist()
    df.loc[idx, 'predictions'] = str([round(i,3) for i in pred])
    df.loc[idx, 'pred'] = res
    print (row['img_path'])
    
classes = sorted(pd.unique(df.labels))
k = []
for idx, row in df.iterrows():
    img = image.load_img(row['img_path'], target_size=(size,size))
    img = image.img_to_array(img)
    img = preprocess_input(np.expand_dims(img, axis=0))
    pred = m.predict(img)
    res = classes[np.argmax(pred)]
    pred = pred[0].tolist()
    df.loc[idx, 'predictions'] = str([round(i,3) for i in pred])
    df.loc[idx, 'pred'] = res
    print (row['img_path'])
    
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy_score(df['labels'].tolist(), df['pred'].tolist())
df.shape
print(classification_report(df['labels'].tolist(), df['pred'].tolist(), target_names=classes))
df.to_csv('/media/train_hdd2/Macys_phase3/art_testset/art_style_testset_inp_valid_reduced_7cls_cropped_images_output_best_model_v2.csv', index=False)
from pandas_ml import ConfusionMatrix
ConfusionMatrix(df['labels'].tolist(), df['pred'].tolist())
