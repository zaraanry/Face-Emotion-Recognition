#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import json
import string
import os
import shutil
import uuid
from captcha.image import ImageCaptcha

import itertools

import os
import cv2
import numpy as np
from random import random

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt


def _gen_captcha(img_dir, num_of_letters, num_of_repetition, width, height):
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    image = ImageCaptcha(width=width, height=height)

    for counter in range(num_of_repetition):
        print('generating %d/%d' % (counter+1, num_of_repetition))
        for i in itertools.permutations([str(c) for c in range(10)], num_of_letters):
            captcha = ''.join(i)
            fn = os.path.join(img_dir, '%s_%s.png' % (captcha, uuid.uuid4()))
            image.write(captcha, fn)


def gen_dataset(path, num_of_repetition, num_of_letters, width, height):
    _gen_captcha(os.path.join(path, 'data'), num_of_letters, num_of_repetition, width, height)
    print('Finished Data Generation')


# ======================================================================================================================
# ======================================================================================================================


BATCH_SIZE = 128
NUM_OF_LETTERS = 5
EPOCHS = 30
IMG_ROW, IMG_COLS = 50, 135

# Non-configs
NUM_OF_CLASSES = NUM_OF_LETTERS * 10
PATH = os.getcwd()
DATA_PATH = os.path.join(PATH, 'train')


def load_data(path, test_split=0.1):
    print 'loading dataset...'
    y_train = []
    y_test = []
    x_train = []
    x_test = []

    # r=root, d=directories, f = files
    counter = 0
    for r, d, f in os.walk(path):
        for fl in f:
            if '.png' in fl:
                counter += 1
                label = np.zeros((NUM_OF_CLASSES, 1))
                for i in range(NUM_OF_LETTERS):
                    label[int(fl[i]) + i*10] = 1

                img = cv2.imread(os.path.join(r, fl))
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # img = np.reshape(img, (50, 135, 1))

                if random() < test_split:
                    y_test.append(label)
                    x_test.append(img)
                else:
                    y_train.append(label)
                    x_train.append(img)

    print('dataset size:', counter, '(train=%d, test=%d)' % (len(y_train), len(y_test)))
    return np.array(x_train), np.array(y_train)[:, :, 0], np.array(x_test), np.array(y_test)[:, :, 0]


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print('Generating Dataset')
        gen_dataset(DATA_PATH, 1, NUM_OF_LETTERS, IMG_COLS, IMG_ROW)

    x_train, y_train, x_test, y_test = load_data(DATA_PATH)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    save_dir = os.path.join(PATH, 'saved_models')
    model_name = 'keras_cifar10_trained_model.h5'

    # Create Model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=x_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(48, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(NUM_OF_CLASSES, activation='sigmoid'))

    # initiate Adam optimizer
    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.99, beta_2=0.9999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1,
              validation_data=(x_test, y_test))

    # Plot training & validation accurac#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import json
import string
import os
import shutil
import uuid
from captcha.image import ImageCaptcha

import itertools

import os
import cv2
import numpy as np
from random import random

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt


def _gen_captcha(img_dir, num_of_letters, num_of_repetition, width, height):
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    image = ImageCaptcha(width=width, height=height)

    for counter in range(num_of_repetition):
        print('generating %d/%d' % (counter+1, num_of_repetition))
        for i in itertools.permutations([str(c) for c in range(10)], num_of_letters):
            captcha = ''.join(i)
            fn = os.path.join(img_dir, '%s_%s.png' % (captcha, uuid.uuid4()))
            image.write(captcha, fn)


def gen_dataset(path, num_of_repetition, num_of_letters, width, height):
    _gen_captcha(os.path.join(path, 'data'), num_of_letters, num_of_repetition, width, height)
    print('Finished Data Generation')


# ======================================================================================================================
# ======================================================================================================================


BATCH_SIZE = 128
NUM_OF_LETTERS = 5
EPOCHS = 30
IMG_ROW, IMG_COLS = 50, 135

# Non-configs
NUM_OF_CLASSES = NUM_OF_LETTERS * 10
PATH = os.getcwd()
DATA_PATH = os.path.join(PATH, 'train')


def load_data(path, test_split=0.1):
    print 'loading dataset...'
    y_train = []
    y_test = []
    x_train = []
    x_test = []

    # r=root, d=directories, f = files
    counter = 0
    for r, d, f in os.walk(path):
        for fl in f:
            if '.png' in fl:
                counter += 1
                label = np.zeros((NUM_OF_CLASSES, 1))
                for i in range(NUM_OF_LETTERS):
                    label[int(fl[i]) + i*10] = 1

                img = cv2.imread(os.path.join(r, fl))
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # img = np.reshape(img, (50, 135, 1))

                if random() < test_split:
                    y_test.append(label)
                    x_test.append(img)
                else:
                    y_train.append(label)
                    x_train.append(img)

    print('dataset size:', counter, '(train=%d, test=%d)' % (len(y_train), len(y_test)))
    return np.array(x_train), np.array(y_train)[:, :, 0], np.array(x_test), np.array(y_test)[:, :, 0]


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print('Generating Dataset')
        gen_dataset(DATA_PATH, 1, NUM_OF_LETTERS, IMG_COLS, IMG_ROW)

    x_train, y_train, x_test, y_test = load_data(DATA_PATH)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    save_dir = os.path.join(PATH, 'saved_models')
    model_name = 'keras_cifar10_trained_model.h5'

    # Create Model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=x_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(48, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(NUM_OF_CLASSES, activation='sigmoid'))

    # initiate Adam optimizer
    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.99, beta_2=0.9999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1,
              validation_data=(x_test, y_test))

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score trained model.
    scores = model.evaluate(x_train, y_train, verbose=1)
    print('Train loss:     %.2f' % scores[0])
    print('Train accuracy: %.2f' % (scores[1]*100.))

    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:     %.2f' % scores[0])
    print('Test accuracy: %.2f' % (scores[1]*100.))#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import json
import string
import os
import shutil
import uuid
from captcha.image import ImageCaptcha

import itertools

import os
import cv2
import numpy as np
from random import random

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt


def _gen_captcha(img_dir, num_of_letters, num_of_repetition, width, height):
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    image = ImageCaptcha(width=width, height=height)

    for counter in range(num_of_repetition):
        print('generating %d/%d' % (counter+1, num_of_repetition))
        for i in itertools.permutations([str(c) for c in range(10)], num_of_letters):
            captcha = ''.join(i)
            fn = os.path.join(img_dir, '%s_%s.png' % (captcha, uuid.uuid4()))
            image.write(captcha, fn)


def gen_dataset(path, num_of_repetition, num_of_letters, width, height):
    _gen_captcha(os.path.join(path, 'data'), num_of_letters, num_of_repetition, width, height)
    print('Finished Data Generation')


# ======================================================================================================================
# ======================================================================================================================


BATCH_SIZE = 128
NUM_OF_LETTERS = 5
EPOCHS = 30
IMG_ROW, IMG_COLS = 50, 135

# Non-configs
NUM_OF_CLASSES = NUM_OF_LETTERS * 10
PATH = os.getcwd()
DATA_PATH = os.path.join(PATH, 'train')


def load_data(path, test_split=0.1):
    print 'loading dataset...'
    y_train = []
    y_test = []
    x_train = []
    x_test = []

    # r=root, d=directories, f = files
    counter = 0
    for r, d, f in os.walk(path):
        for fl in f:
            if '.png' in fl:
                counter += 1
                label = np.zeros((NUM_OF_CLASSES, 1))
                for i in range(NUM_OF_LETTERS):
                    label[int(fl[i]) + i*10] = 1

                img = cv2.imread(os.path.join(r, fl))
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # img = np.reshape(img, (50, 135, 1))

                if random() < test_split:
                    y_test.append(label)
                    x_test.append(img)
                else:
                    y_train.append(label)
                    x_train.append(img)

    print('dataset size:', counter, '(train=%d, test=%d)' % (len(y_train), len(y_test)))
    return np.array(x_train), np.array(y_train)[:, :, 0], np.array(x_test), np.array(y_test)[:, :, 0]


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print('Generating Dataset')
        gen_dataset(DATA_PATH, 1, NUM_OF_LETTERS, IMG_COLS, IMG_ROW)

    x_train, y_train, x_test, y_test = load_data(DATA_PATH)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    save_dir = os.path.join(PATH, 'saved_models')
    model_name = 'keras_cifar10_trained_model.h5'

    # Create Model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=x_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(48, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(NUM_OF_CLASSES, activation='sigmoid'))

    # initiate Adam optimizer
    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.99, beta_2=0.9999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1,
              validation_data=(x_test, y_test))

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score trained model.
    scores = model.evaluate(x_train, y_train, verbose=1)
    print('Train loss:     %.2f' % scores[0])
    print('Train accuracy: %.2f' % (scores[1]*100.))

    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:     %.2f' % scores[0])
    print('Test accuracy: %.2f' % (scores[1]*100.))y values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score trained model.
    scores = model.evaluate(x_train, y_train, verbose=1)
    print('Train loss:     %.2f' % scores[0])
    print('Train accuracy: %.2f' % (scores[1]*100.))

    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:     %.2f' % scores[0])
    print('Test accuracy: %.2f' % (scores[1]*100.))import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

def load_image(image_path, grayscale=False, target_size=None):
    pil_image = image.load_img(image_path, grayscale, target_size)
    return image.img_to_array(pil_image)

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=0.5, thickness=1):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

def get_colors(num_classes):
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    colors = np.asarray(colors) * 255
    return colors
