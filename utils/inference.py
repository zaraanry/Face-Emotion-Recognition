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
    paddedw = image_shape[1] + border
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
    plt.show()import numpy as np
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
    paddedw = image_shape[1] + border
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
    plt.show()import numpy as np
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
    paddedw = image_shape[1] + border
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
    plt.show()import numpy as np
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
    paddedw = image_shape[1] + border
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
    plt.show()import cv2
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

    cv2.imwrite('guided_gradCAM.jpg', guided_gradCAM)import cv2
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