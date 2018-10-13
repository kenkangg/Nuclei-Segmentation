

import numpy as np
from imageio import imread
from os import listdir, walk
import os

from scipy import ndimage
from skimage.measure import find_contours
from skimage.transform import resize

from skimage import color
from skimage import io


def read_images(path):
    """ Read Images and Masks from Path

        Arguments:
            path

        Return:
            dictionary of image, indexed by image name
    """

    images = {}

    count = 0

    for root,dirs,files in os.walk(path):

        if 'images' in dirs:
            for img_name in listdir(root + '/images'):

                if count % 500 == 0:
                    print("Images Read: {}".format(count))
                count += 1

                # Read Images
                img_path = root + '/images/%s' % img_name
                if img_path == '':
                    raise Exception("No Images found in directory")
                img = imread(img_path)[:,:,:3]
                images[img_name] = img
    print("Completed processing {} images".format(count))

    return images


def read_masks(path):
    """ Read Images and Masks from Path

        Arguments:
            path

        Return:
            dictionary of masks, indexed by corresponding image name


    """

    masks = {}

    count = 0

    for root,dirs,files in os.walk(path):
        if 'images' in dirs:
            for img_name in listdir(root + '/images'):

                if count % 500 == 0:
                    print("Masks Read: {}".format(count))
                count += 1

                # Read Masks
                submasks = []
                for mask_name in listdir(root + '/masks'):
                    mask_path = root + '/masks/{}'.format(mask_name)
                    if mask_path == '':
                        raise Exception("No Masks found in directory")
                    submasks.append(imread(mask_path))
                masks[img_name] = submasks
    print("Completed processing {} masks".format(count))
    return masks


def read_test_images(path):
    """ Read Images and Masks from Path

        Arguments:
            path

        Return:
            dictionary of image names, each containing dictionary of {image,masks}.


    """

    images = {}
    sizes = []
    img_names = []

    count = 0

    for root,dirs,files in os.walk(path):

        if 'images' in dirs:
            for img_name in listdir(root + '/images'):

                if count % 500 == 0:
                    print("Read {} image".format(count))
                count += 1

                # Read Images
                img_path = root + '/images/%s' % img_name
                if img_path == '':
                    raise Exception("No Images found in directory")
                img = imread(img_path)
                if img.shape[-1] != 3:
                    try:
                        img = img[:,:,:3]
                    except IndexError:
                        img = np.expand_dims(img,axis=-1)
                        img = np.repeat(img,3, axis=-1)
                images[img_name] = img
                sizes.append(img.shape)
                img_names.append(img_name)
    print("Completed processing {} images".format(count))

    return images, sizes, img_names

def get_train_images(images, gray=False, shape=(256,256,3)):
    return uniform_reshape(images, shape, gray)

def get_train_labels(submasks):
    masks = {}
    for key in submasks.keys():
        masks[key] = combine_output_mask(submasks[key])
    return uniform_reshape(masks, (256,256))

def get_test_masks(submasks):
    masks = {}
    for key in submasks.keys():
        masks[key] = combine_test_mask(submasks[key])
    return uniform_reshape(masks, (256,256))


def combine_output_mask(masks):
    """ Combines multiple masks into a single mask

    Arguments:
        masks: np.array of masks for given image

    Return:
        mask: single mask of joined submasks
    """
    return np.amax(masks, axis=0)

def combine_test_mask(masks):
    """ Combines multiple masks into a single mask with each mask having a unique value

    Arguments:
        masks: np.array of masks for given image

    Return:
        mask: single mask of joined submasks
    """
    mask = np.zeros((masks[0].shape))
    for i in range(len(masks)):
        mask = np.add(mask, (masks[i] / 255) * (i + 1))

    return mask

def create_train_data(images, shape):
    """ Return Numpy Array of Images

        Arguments:
            images: list of images(np.array)
            shape: tuple of single image sample size

        return:
            train_data: np.array of resized images
    """
    train = []
    for img in images:
        reshaped = resize(img, shape)
        data.append(reshaped)

    train_data = np.stack(train, axis=0)
    return data

def create_train_labels(images):
    label_data = {}
    weight_map = {}
    count = 0
    for key in images:
        if count % 50 == 0:
             print(count)

        count+= 1
        final_mask = combine_mask(key, masks[key])
        label_data[key] = final_mask
    return final_mask





def uniform_reshape(images, shape, gray=False):

    """ Reshape data from dictionary of matrices to a single multidimensional numpy array

    Arguments:
    data = Dictionary of numpy arrays representing an image.
    shape = desired shape of output (h,w,c)

    Return:
    numpy array of shape (num_images, h, w, c)
    """
    data = []
    count = 0
    for key in images.keys():
        if count % 500 == 0:
            print(count)
        count += 1
        if gray:
            image = color.rgb2gray(images[key])
        else:
            image = images[key]
        reshaped = resize(image, shape)
        data.append(reshaped)


    data = np.stack(data, axis=0)
    return data

def get_image_ordering(images):
    ordering = []
    for key in images.keys():
        ordering.append(key)
    return ordering

def preserve_original_sizes(images):
    original_image_sizes = {}
    for key in images.keys():
        w,h,c = images[key].shape
        original_image_sizes[key] = (w,h,3)
    return original_image_sizes

def calculate_unet_background_weight(merged_mask, masks, w0=10, q=5,):
    weight = np.zeros(merged_mask.shape)
    # calculate weight for important pixels
    distances = np.array([ndimage.distance_transform_edt(imread(m)==0) for m in masks])
    shortest_dist = np.sort(distances, axis=0)
    # distance to the border of the nearest cell
    d1 = shortest_dist[0]
    # distance to the border of the second nearest cell
    d2 = shortest_dist[1] if len(shortest_dist) > 1 else np.zeros(d1.shape)

    weight = w0 * np.exp(-(d1+d2)**2/(2*q**2)).astype(np.float32)
    weight = 1 + (merged_mask == 0) * weight
    return weight
