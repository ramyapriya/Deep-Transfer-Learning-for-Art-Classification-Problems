from keras.models import load_model
import pandas as pd
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt

# Config
size = 224
channels = 3

model_path = sys.argv[1]
weights_path = sys.argv[2]
inp_csv = sys.argv[3]
op_csv = sys.argv[4]

# Load model
m = load_model(model_path)
m.load_weights(weights_path)

df = pd.read_csv(inp_csv)
classes = sorted(pd.unique(df.labels))


k = []
for idx, row in df.iterrows():
    img = image.load_img(row['img_path'], target_size=(size, size))
    img = image.img_to_array(img)
    img = preprocess_input(np.expand_dims(img, axis=0))
    pred = m.predict(img)
    res = classes[np.argmax(pred)]
    pred = pred[0].tolist()
    df.loc[idx, 'predictions'] = str([round(i, 3) for i in pred])
    df.loc[idx, 'pred'] = res

df.to_csv(op_csv, index=False)

# Metrics
print (accuracy_score(df['labels'].tolist(), df['pred'].tolist()))
print (classification_report(df['labels'].tolist(), df['pred'].tolist(), target_names=classes))
cm = ConfusionMatrix(df['labels'].tolist(), df['pred'].tolist()).to_dataframe()

# Save metrics
cm.to_csv(os.path.join(os.path.dirname(op_csv), 'confusion_matrix.csv'), index=False)
fig = plt.figure()
plt.matshow(cm)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(os.path.dirname(op_csv), 'confusion_matrix.jpg'))
