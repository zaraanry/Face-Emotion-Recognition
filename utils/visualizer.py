import numpy as np
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy.ma as ma
import cv2
from .inference import draw_text

def make_mosaic(images, num_rows, num_cols, border=1, class_names=None):
    num_images = len(images)
    image_shape = images.shape[1:]
    mosaic = ma.masked_all((num_rows * image_shape[0] + (num_rows - 1) * border,
                            num_cols * image_shape[1] + (num_cols - 1) * border),
                            dtype=np.float32)
    paddedh = image_shape[0] + border
    paddedw = image_shape[1] + border#!/usr/bin/env python
# coding: utf-8


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
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input
import matplotlib.pyplot as plt


def _gen_captcha(img_dir, num_of_letters, num_of_repetition, width, height):
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    image = ImageCaptcha(width=width, height=height)

    for counter in range(num_of_repetition):
        print('generating %d/%d' % (counter + 1, num_of_repetition))
        for i in itertools.permutations([str(c) for c in range(10)], num_of_letters):
            captcha = ''.join(i)
            fn = os.path.join(img_dir, '%s_%s.png' % (captcha, uuid.uuid4()))
            image.write(captcha, fn)


def gen_dataset(path, num_of_repetition, num_of_letters, width, height):
    _gen_captcha(os.path.join(path, 'data'), num_of_letters, num_of_repetition, width, height)
    print('Finished Data Generation')


BATCH_SIZE = 128
NUM_OF_LETTERS = 5
EPOCHS = 35
IMG_ROW, IMG_COLS = 50, 135


# Non-configs
PATH = os.getcwd()
DATA_PATH = os.path.join(PATH, 'train')


def load_data(path, test_split=0.1):
    print ('loading dataset...')
    y_train = []
    y_test = []
    x_train = []
    x_test = []

    # r=root, d=directories, f = files
    counter = 0
    for r, d, f in os.walk(path):
        for fl in f:
            if '.png' in fl:
                flr = fl.split('_')[0]
                counter += 1
                label = np.zeros((NUM_OF_LETTERS, 10))
                for i in range(NUM_OF_LETTERS):
                    label[i, int(flr[i])] = 1

                img = cv2.imread(os.path.join(r, fl))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (int(135/2), int(50/2)), interpolation=cv2.INTER_AREA)
                img = np.reshape(img, (img.shape[0], img.shape[1], 1))

                if random() < test_split:
                    y_test.append(label)
                    x_test.append(img)
                else:
                    y_train.append(label)
                    x_train.append(img)

    print('dataset size:', counter, '(train=%d, test=%d)' % (len(y_train), len(y_test)))
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


if not os.path.exists(DATA_PATH):
    print('Generating Dataset')
    gen_dataset(DATA_PATH, 8 , NUM_OF_LETTERS, IMG_COLS, IMG_ROW)


x_train, y_train, x_test, y_test = load_data(DATA_PATH)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


s_train = []
s_test = []
for i in range(NUM_OF_LETTERS):
    s_train.append(y_train[:, i, :])
    s_test.append(y_test[:, i, :])


save_dir = os.path.join(PATH, 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'


input_layer = Input((25, 67, 1))
x = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Dropout(0.3)(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)

out = [Dense(10, name='digit%d' % i, activation='softmax')(x) for i in range(NUM_OF_LETTERS)]
model = Model(inputs=input_layer, outputs=out)

# initiate Adam optimizer
opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.99, beta_2=0.9999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()    


digit_acc = [[] for _ in range(NUM_OF_LETTERS)]
val_digit_acc = [[] for _ in range(NUM_OF_LETTERS)]
loss = []
val_loss = []


def plot_diagram(digit_acc_now, val_digit_acc_now, loss_now, val_loss_now):
    global digit_acc, val_digit_acc, loss, val_loss
    for i in range(NUM_OF_LETTERS):
        digit_acc[i].extend(digit_acc_now[i])
        val_digit_acc[i].extend(val_digit_acc_now[i])
    loss.extend(loss_now)
    val_loss.extend(val_loss_now)
    
    
    for i in range(NUM_OF_LETTERS):
        plt.plot(digit_acc[i], label='digit%d_acc' % i)
        plt.plot(val_digit_acc[i], label='val_digit%d_acc' % i)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    # Plot training & validation loss values
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


history = model.fit(x_train, s_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=(x_test, s_test))


plot_diagram(
    [history.history['digit%d_acc' % i] for i in range(NUM_OF_LETTERS)],
    [history.history['val_digit%d_acc' % i] for i in range(NUM_OF_LETTERS)],
    history.history['loss'],
    history.history['val_loss'],
)


# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


# Score trained model.
scores = model.evaluate(x_train, s_train, verbose=1)
print('Train loss:     %.2f' % np.mean(scores[0:5]))
acc = 1.
for i in range(5):
    acc *= scores[6+i]
print('Train accuracy: %.2f' % (acc * 100.))


scores = model.evaluate(x_test, s_test, verbose=1)
print('Test loss:     %.2f' % np.mean(scores[0:5]))
acc = 1.
for i in range(5):
    acc *= scores[6+i]
print('Test accuracy: %.2f' % (acc * 100.))


    for image_arg in range(num_images):
        row = int(np.floor(image_arg / num_cols))
        col = image_arg % num_cols
        image = np.squeeze(images[image_arg])
        image_shape = image.shape
        mosaic[row * paddedh:row * paddedh + image_shape[0],
               col * paddedw:col * paddedw + image_shape[1]] = image
    return mosaic

def make_mosaic_v2(images, num_mosaic_rows=None,
                num_mosaic_cols=None, border=1):
    images = np.squeeze(images)
    num_images, image_pixels_rows, image_pixels_cols = images.shape
    if num_mosaic_rows is None and num_mosaic_cols is None:
        box_size = int(np.ceil(np.sqrt(num_images)))
        num_mosaic_rows = num_mosaic_cols = box_size
    num_mosaic_pixel_rows = num_mosaic_rows * (image_pixels_rows + border)
    num_mosaic_pixel_cols = num_mosaic_cols * (image_pixels_cols + border)
    mosaic = np.empty(shape=(num_mosaic_pixel_rows, num_mosaic_pixel_cols))
    mosaic_col_arg = 0
    mosaic_row_arg = 0
    for image_arg in range(num_images):
        if image_arg % num_mosaic_cols == 0 and image_arg != 0:
            mosaic_col_arg = mosaic_col_arg + 1
            mosaic_row_arg = 0
        x0 = image_pixels_cols * (mosaic_row_arg)
        x1 = image_pixels_cols * (mosaic_row_arg + 1)
        y0 = image_pixels_rows * (mosaic_col_arg)
        y1 = image_pixels_rows * (mosaic_col_arg + 1)
        image = images[image_arg]
        mosaic[y0:y1, x0:x1] = image
        mosaic_row_arg = mosaic_row_arg + 1
    return mosaic

def pretty_imshow(axis, data, vmin=None, vmax=None, cmap=None):
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    cax = None
    divider = make_axes_locatable(axis)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    image = axis.imshow(data, vmin=vmin, vmax=vmax,
                        interpolation='nearest', cmap=cmap)
    plt.colorbar(image, cax=cax)

def normal_imshow(axis, data, vmin=None, vmax=None,
                        cmap=None, axis_off=True):
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    image = axis.imshow(data, vmin=vmin, vmax=vmax,
                        interpolation='nearest', cmap=cmap)
    if axis_off:
        plt.axis('off')
    return image

def display_image(face, class_vector=None,
                    class_decoder=None, pretty=False):
    if class_vector is not None and class_decoder is None:
        raise Exception('Provide class decoder')
    face = np.squeeze(face)
    color_map = None
    if len(face.shape) < 3:
        color_map = 'gray'
    plt.figure()
    if class_vector is not None:
        class_arg = np.argmax(class_vector)
        class_name = class_decoder[class_arg]
        plt.title(class_name)
    if pretty:
        pretty_imshow(plt.gca(), face, cmap=color_map)
    else:
        plt.imshow(face, color_map)

def draw_mosaic(data, num_rows, num_cols, class_vectors=None,
                            class_decoder=None, cmap='gray'):

    if class_vectors is not None and class_decoder is None:
        raise Exception('Provide class decoder')

    figure, axis_array = plt.subplots(num_rows, num_cols)
    figure.set_size_inches(8, 8, forward=True)
    titles = []
    if class_vectors is not None:
        for vector_arg in range(len(class_vectors)):
            class_arg = np.argmax(class_vectors[vector_arg])
            class_name = class_decoder[class_arg]
            titles.append(class_name)

    image_arg = 0
    for row_arg in range(num_rows):
        for col_arg in range(num_cols):
            image = data[image_arg]
            image = np.squeeze(image)
            axis_array[row_arg, col_arg].axis('off')
            axis_array[row_arg, col_arg].imshow(image, cmap=cmap)
            axis_array[row_arg, col_arg].set_title(titles[image_arg])
            image_arg = image_arg + 1
    plt.tight_layout()

if __name__ == '__main__':
    from utils.utils import get_labels
    from keras.models import load_model
    import pickle

    dataset_name = 'fer2013'
    class_decoder = get_labels(dataset_name)
    faces = pickle.load(open('faces.pkl', 'rb'))
    emotions = pickle.load(open('emotions.pkl', 'rb'))
    pretty_imshow(plt.gca(), make_mosaic(faces[:4], 2, 2), cmap='gray')
    plt.show()

    model = load_model('../trained_models/emotion_models/simple_CNN.985-0.66.hdf5')
    conv1_weights = model.layers[2].get_weights()
    kernel_conv1_weights = conv1_weights[0]
    kernel_conv1_weights = np.squeeze(kernel_conv1_weights)
    kernel_conv1_weights = np.rollaxis(kernel_conv1_weights, 2, 0)
    kernel_conv1_weights = np.expand_dims(kernel_conv1_weights, -1)
    num_kernels = kernel_conv1_weights.shape[0]
    box_size = int(np.ceil(np.sqrt(num_kernels)))
    print('Box size:', box_size)

    print('Kernel shape', kernel_conv1_weights.shape)
    plt.figure(figsize=(15, 15))
    plt.title('conv1 weights')
    pretty_imshow(plt.gca(),
            make_mosaic(kernel_conv1_weights, box_size, box_size),
            cmap=cm.binary)
    plt.show()
