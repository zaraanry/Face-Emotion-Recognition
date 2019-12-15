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
import h5py
import keras
import keras.backend as K
from keras.layers.core import Lambda
from keras.models import Sequential
from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from .preprocessor import preprocess_input

def reset_optimizer_weights(model_filename):
    model = h5py.File(model_filename, 'r+')
    del model['optimizer_weights']
    model.close()


def target_category_loss(x, category_index, num_classes):
    return tf.multiply(x, K.one_hot([category_index], num_classes))


def target_category_loss_output_shape(input_shape):
    return input_shape


def normalize(x):

    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def load_image(image_array):
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_array


def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, gradient):
            dtype = op.inputs[0].dtype
            guided_gradient = (gradient * tf.cast(gradient > 0., dtype) *
                               tf.cast(op.inputs[0] > 0., dtype))
            return guided_gradient


def compile_saliency_function(model, activation_layer='conv2d_7'):
    input_image = model.input
    layer_output = model.get_layer(activation_layer).output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_image)[0]
    return K.function([input_image, K.learning_phase()], [saliency])


def modify_backprop(model, name, task):
    graph = tf.get_default_graph()
    with graph.gradient_override_map({'Relu': name}):

        activation_layers = [layer for layer in model.layers
                             if hasattr(layer, 'activation')]

        for layer in activation_layers:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        if task == 'gender':
            model_path = '../trained_models/gender_models/gender_mini_XCEPTION.21-0.95.hdf5'
        elif task == 'emotion':
            model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
        new_model = load_model(model_path, compile=False)
    return new_model


def deprocess_image(x):

    if np.ndim(x) > 3:
        x = np.squeeze(x)
    x = x - x.mean()
    x = x / (x.std() + 1e-5)
    x = x * 0.1

    # clip to [0, 1]
    x = x + 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x = x * 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def compile_gradient_function(input_model, category_index, layer_name):
    model = Sequential()
    model.add(input_model)

    num_classes = model.output_shape[1]
    target_layer = lambda x: target_category_loss(x, category_index, num_classes)
    model.add(Lambda(target_layer,
                     output_shape = target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)
    conv_output = model.layers[0].get_layer(layer_name).output
    gradients = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input, K.learning_phase()],
                                                    [conv_output, gradients])
    return gradient_function

def calculate_gradient_weighted_CAM(gradient_function, image):
    output, evaluated_gradients = gradient_function([image, False])
    output, evaluated_gradients = output[0, :], evaluated_gradients[0, :, :, :]
    weights = np.mean(evaluated_gradients, axis = (0, 1))
    CAM = np.ones(output.shape[0 : 2], dtype=np.float32)
    for weight_arg, weight in enumerate(weights):
        CAM = CAM + (weight * output[:, :, weight_arg])
    CAM = cv2.resize(CAM, (64, 64))
    CAM = np.maximum(CAM, 0)
    heatmap = CAM / np.max(CAM)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image = image - np.min(image)
    image = np.minimum(image, 255)

    CAM = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    CAM = np.float32(CAM) + np.float32(image)
    CAM = 255 * CAM / np.max(CAM)
    return np.uint8(CAM), heatmap

def calculate_guided_gradient_CAM(preprocessed_input, gradient_function, saliency_function):
    CAM, heatmap = calculate_gradient_weighted_CAM(gradient_function, preprocessed_input)
    saliency = saliency_function([preprocessed_input, 0])
    gradCAM = saliency[0] * heatmap[..., np.newaxis]
    #return deprocess_image(gradCAM)
    return deprocess_image(saliency[0])
    #return saliency[0]

def calculate_guided_gradient_CAM_v2(preprocessed_input, gradient_function,
                                    saliency_function, target_size=(128, 128)):
    CAM, heatmap = calculate_gradient_weighted_CAM(gradient_function, preprocessed_input)
    heatmap = np.squeeze(heatmap)
    heatmap = cv2.resize(heatmap.astype('uint8'), target_size)
    saliency = saliency_function([preprocessed_input, 0])
    saliency = np.squeeze(saliency[0])
    saliency = cv2.resize(saliency.astype('uint8'), target_size)
    gradCAM = saliency * heatmap
    gradCAM =  deprocess_image(gradCAM)
    return np.expand_dims(gradCAM, -1)


if __name__ == '__main__':
    import pickle
    faces = pickle.load(open('faces.pkl','rb'))
    face = faces[0]
    model_filename = '../../trained_models/emotion_models/mini_XCEPTION.523-0.65.hdf5'
    #reset_optimizer_weights(model_filename)
    model = load_model(model_filename)

    preprocessed_input = load_image(face)
    predictions = model.predict(preprocessed_input)
    predicted_class = np.argmax(predictions)
    gradient_function = compile_gradient_function(model, predicted_class, 'conv2d_6')
    register_gradient()
    guided_model = modify_backprop(model, 'GuidedBackProp')
    saliency_function = compile_saliency_function(guided_model)
    guided_gradCAM = calculate_guided_gradient_CAM(preprocessed_input,
                                gradient_function, saliency_function)

    cv2.imwrite('guided_gradCAM.jpg', guided_gradCAM)
