import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MultiLabelBinarizer

labels = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]

mlb = MultiLabelBinarizer()
mlb.fit(labels)
mlb.classes_


def CNN():
    # inisiasi CNN
    model = Sequential()

    # Convolution dan Max Pooling
    model.add(Conv2D(35, (3, 3), input_shape=(64, 64, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 2x2 is optimal

    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattening
    model.add(Flatten())

    # Dense atau Full Connection
    model.add(Dropout(0.5))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=26, activation="softmax"))

    return model


SibiKlasifikasi = CNN()
SibiKlasifikasi.load_weights("model\model_sign_e75.h5")


def getLabel(label, model_label):
    for i, x in enumerate(model_label):
        if x >= 0.6:
            return label[i]

    return "Gambar bukan abjad SIBI"


def Classification(img_input, path):
    dataset_path = path
    data_test = os.listdir(dataset_path)
    hasil = data_test
    for x in range(len(hasil)):
        if hasil[x] == img_input:
            img = cv2.imread(dataset_path + "/" + hasil[x])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resize = cv2.resize(img, dsize=(64, 64))
            img = np.expand_dims(resize, axis=0)
            predict = SibiKlasifikasi.predict(img)
            label = (hasil[x], getLabel(mlb.classes_, predict[0]))
            hasil.append(label)
            return hasil[x][0]

    return "-"
