# code for MSP
# To read audio file in Python
import wave
from scipy.io import wavfile
import os
# --------------code to plot audio file
import matplotlib.pyplot as plt
import numpy as np
#code to translate audio files into 1 second small chunks
import simpleaudio as sa
from pydub import AudioSegment
from pydub.utils import make_chunks
import os
path = "D:\\Learn_ML\\IEEE_NLP\\midlands_english_female"
i=0
for file in os.listdir(path):
    if i==3000:
        break
    # Check whether file is in text format or not
    if file.endswith(".wav"):
       myaudio = AudioSegment.from_file(path+'\\'+file)
       chunk_length_ms = 1000
       chunks = make_chunks(myaudio, chunk_length_ms)
       for j, chunk in enumerate(chunks):
             chunk_name = "D:\\Learn_ML\\IEEE_NLP\\midlands_english_female\\English_chunk_f\\English_chunk_f{0}.wav".format(i)
             print("exporting", chunk_name)
             chunk.export(chunk_name, format="wav")
             i=i+1

import os  # all below imported packeages are important
import cv2
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from os import listdir
from os import listdir
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import json
from skimage import color
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Used to avoid printing of Warnings
folder_dir = "D:\\Learn_ML\\IEEE_NLP\\English_train\\English_chunk"

import librosa
import librosa.display
n_fft = 2048
hop_length = 512
n_mels = 128
import numpy as np
import matplotlib.pyplot as plt
folder_dir1 = 'D:\\Learn_ML\\IEEE_NLP\\spectogram'

audiolist1 = []
filelist1 = []
labellist1 = []
featurelist1 = []
faudiolist1 = []

for files in os.listdir(folder_dir1):

    # check if the image ends with png
    if (files.endswith(".wav")):
        x, sr = librosa.load(folder_dir1 + '\\' + files)
        S = librosa.feature.melspectrogram(x, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)
        filelist1.append(S_DB)
for files in os.listdir(folder_dir1):

    # check if the image ends with png
    if (files.endswith(".wav")):
        s = files
        result = s.find('_')
        label = s[0:result]
        labellist1.append(label)
        print(label)

imagelist = []
normlist = []

i = 0
while (i < 10663):  # To take only non-empty images
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(filelist1[i], (224, 224),
                      interpolation=cv2.INTER_NEAREST)  # The INTER_NEAREST method uses the nearest neighbor concept for interpolation. This is one of the simplest methods, using only one neighboring pixel from the image for interpolation.
    norm_image = cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    imagelist.append(norm_image)
    i = i + 1
yy = np.array(labellist1)  # convert list into array
xx = np.array(imagelist)
with open('D:\\Learn_ML\\IEEE_NLP\\xx.npy', 'wb') as f:
    np.save(f, xx)

with open('D:\\Learn_ML\\IEEE_NLP\\yy.npy', 'wb') as f:
    np.save(f, yy)

x = np.load('D:\\Learn_ML\\IEEE_NLP\\xx.npy')
y = np.load('D:\\Learn_ML\\IEEE_NLP\\yy.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0, stratify=y)
x_train1 = np.expand_dims(x_train, axis=3)
x_test1 = np.expand_dims(x_test, axis=3)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y_train)
integer_encoded1 = label_encoder.fit_transform(y_test)

print(integer_encoded)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
integer_encoded1 = integer_encoded1.reshape(len(integer_encoded1), 1)
y_train1 = onehot_encoder.fit_transform(integer_encoded)
y_test1 = onehot_encoder.fit_transform(integer_encoded1)

print(y_train1)

from numpy import argmax

# invert first example
inverted = label_encoder.inverse_transform([argmax(y_train1[0, :])])
inverted1 = label_encoder.inverse_transform([argmax(y_test1[0, :])])
print(inverted)
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
# create model
model = Sequential()
model.add(Conv2D(input_shape=(224,224,1),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=3, activation='softmax'))

model.compile(
          optimizer='adam',
          loss='categorical_crossentropy',
          metrics=['accuracy'],
              ) 
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=3, restore_best_weights=True, min_delta=0.001)
mc = ModelCheckpoint('D:\\Learn_ML\\IEEE_NLP\\IEEE_model_mobilenetv2_final2.hdf5', monitor='val_loss', mode='min',
                     verbose=1, save_best_only=True)

hist = model.fit(x_train1, y_train1, epochs=15, validation_split=0.2, batch_size=32, callbacks=[es, mc])
scores = model.evaluate(x_train1, y_train1, verbose=0)
from keras.models import load_model

saved_model = load_model('D:\\Learn_ML\\IEEE_NLP\\IEEE_model_mobilenetv2_final2.hdf5')
saved_model = model
_, train_acc = saved_model.evaluate(x_train1, y_train1, verbose=0)  # No progress bar shown
_, test_acc = saved_model.evaluate(x_test1, y_test1, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
saved_model.evaluate(x_test1[0], y_test1[0])
y_pred = saved_model.predict(x_test1)
y_pred1 = np.round(y_pred)
temp = []
for i in y_pred1:
    temp.append(i.argmax(axis=0))  # to extract position where output is 1
temp = list(temp)
temp1 = []
for i in y_test1:
    temp1.append(i.argmax(axis=0))
xx = np.arange(0, 3)
accuracy = accuracy_score(temp, temp1)
print('Accuracy: %f' % accuracy)
x_1 = np.expand_dims(x, axis=3)
y_pred2 = saved_model.predict(x_1)
y_pred_2 = np.round(y_pred2)
temp_1 = []
for i in y_pred_2:
    temp_1.append(i.argmax(axis=0))  # to extract position where output is 1
temp_1 = list(temp_1)
xx= np.arange(0,3)
accuracy = accuracy_score(temp, temp1)
print('Accuracy: %f' % accuracy)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(temp1, temp, labels=xx)
recall = np.diag(cm) / np.sum(cm, axis = 1)
precision = np.diag(cm) / np.sum(cm, axis = 0)
recall1=np.mean(recall)
precision1=np.mean(precision)
f1=(2*recall1*precision1)/(recall1+precision1)
print('Precision: %f'% np.mean(precision))
# recall: tp / (tp + fn)
print('Recall: %f' % np.mean(recall))
print('F1 score: %f' % f1)
fig, px = plt.subplots(figsize=(7.5, 7.5))
px.matshow(cm, cmap=plt.cm.YlOrRd, alpha=0.5)
for m in range(cm.shape[0]):
    for n in range(cm.shape[1]):
        px.text(x=m,y=n,s=cm[m, n], va='center', ha='center', size='xx-large')

# Sets the labels
plt.xlabel('Predictions', fontsize=16)
plt.ylabel('Actuals', fontsize=16)
plt.title('Confusion Matrix', fontsize=15)
plt.show()
#For predicting output from a given audio file
from keras.models import load_model
saved_model = load_model('D:\\Learn_ML\\IEEE_NLP\\IEEE_model_mobilenetv2_final2.hdf5')

myaudio = AudioSegment.from_file('gdrive/My Drive/colab_input/01_0003_01_1.wav') #given audio file
chunk_length_ms = 1000
chunks = make_chunks(myaudio, chunk_length_ms)
i=0
for j, chunk in enumerate(chunks):
             chunk_name = "gdrive/My Drive/colab_input_1/temp_f{0}.wav".format(i)
             print("exporting", chunk_name)
             chunk.export(chunk_name, format="wav")
             i=i+1
folder_dir1='gdrive/My Drive/colab_input_1'

audiolist1 = []
filelist1=[]
labellist1=[]
featurelist1=[]
faudiolist1=[]
imagelist=[]
normlist=[]
for files in os.listdir(folder_dir1):
 
    # check if the image ends with wav
    if (files.endswith(".wav")):
        x , sr = librosa.load(folder_dir1+'/'+files) 
        S = librosa.feature.melspectrogram(x, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)
        img1=cv2.resize(S_DB, (224, 224),interpolation = cv2.INTER_NEAREST)#The INTER_NEAREST method uses the nearest neighbor concept for interpolation. This is one of the simplest methods, using only one neighboring pixel from the image for interpolation.
        norm_image = cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)    
        imagelist.append(norm_image)
xx = np.array(imagelist)
xx_1 = np.expand_dims(xx, axis=3)
y_pred=saved_model.predict(xx_1)
y_pred1=np.round(y_pred)
temp=[]
temp1=[]
for i in y_pred1:
    temp.append(i.argmax(axis=0))# to extract position where output is 1
temp=list(temp)
from collections import Counter
 
def most_frequent(temp):
    occurence_count = Counter(temp)
    return occurence_count.most_common(1)[0][0]
   
f=most_frequent(temp)
if f==0:
  print("English")
elif f==1:
  print("Hindi")
else:
  print("Marathi")
