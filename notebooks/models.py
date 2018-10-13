from model_utils import *
from keras import backend as K
cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))

import numpy as np

import keras
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D, concatenate
from keras.models import Model

from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Activation, Dense, BatchNormalization, Reshape
from keras import losses, optimizers
import tensorflow as tf

from keras.applications.vgg16 import VGG16

from functools import partial

class Unet_Vanilla:
    def __init__(self):
        inputs = Input((256, 256, 3))
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])

        self.model = model

    def compile(self, optimizer=optimizers.Adam(0.001), loss=weighted_binary_crossentropy_loss, metrics=[mean_iou]):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[mean_iou])

    def train(self, x, y, epochs=50, batch_size=32, validation_data = None):
        self.model.fit(x, y, batch_size = batch_size, epochs = epochs, validation_data=validation_data)


class Unet_VGG16:
    def __init__(self):
        inputs2 = Input((256, 256, 3))
        base_model = VGG16(input_tensor=inputs2 ,weights='imagenet', include_top=False)
        outputs = [layer.output for layer in base_model.layers]


        x = base_model.output

        up2 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name='conv1')(x), outputs[17]], axis=3)
        conv2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2')(up2)
        conv2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv3')(conv2)

        up3 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='conv4')(conv2), outputs[13]], axis=3)
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv5')(up3)
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv6')(conv3)

        up4 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='conv7')(conv3), outputs[9]], axis=3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv8')(up4)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv9')(conv4)

        up5 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='conv10')(conv4), outputs[5]], axis=3)
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv11')(up5)
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv12')(conv5)

        up6 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='conv13')(conv5), outputs[2]], axis=3)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv14')(up6)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv15')(conv6)

        conv7 = Conv2D(1, (1, 1), activation='sigmoid',name='conv16')(conv6)

        model2 = Model(inputs=[base_model.input], outputs=[conv7])

        for layer in model2.layers[:19]:
            layer.trainable = False
        for layer in model2.layers[19:]:
            layer.trainable = True

        self.model = model2

    def compile(self, optimizer=optimizers.Adam(0.001), loss=weighted_binary_crossentropy_loss, metrics=[mean_iou]):
        self.model.compile(optimizer=optimzier, loss=loss, metrics=[mean_iou])

    def unfreeze_encoder(self,optimizer=optimizers.Adam(0.0003)):
        for layer in self.model.layers:
            layer.trainable = True
        # sgd = optimizers.SGD(lr=0.0003, decay=1e-6, momentum=0.9, nesterov=True)
        self.compile(optimizer)

    def train(self, x, y, epochs=50, batch_size=32, validation_data = None):
        self.model.fit(x, y, batch_size = 32, epochs = 500, validation_data=validation_data)





class Unet_VGG16_Weighted:
    def __init__(self):
        inputs3 = Input((256, 256, 3))
        base_model = VGG16(input_tensor=inputs3 ,weights='imagenet', include_top=False)
        outputs = [layer.output for layer in base_model.layers]


        x = base_model.output

        up2 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name='conv1')(x), outputs[17]], axis=3)
        conv2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2')(up2)
        conv2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv3')(conv2)

        up3 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='conv4')(conv2), outputs[13]], axis=3)
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv5')(up3)
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv6')(conv3)

        up4 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='conv7')(conv3), outputs[9]], axis=3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv8')(up4)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv9')(conv4)

        up5 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='conv10')(conv4), outputs[5]], axis=3)
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv11')(up5)
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv12')(conv5)

        up6 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='conv13')(conv5), outputs[2]], axis=3)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv14')(up6)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv15')(conv6)

        conv7 = Conv2D(1, (1, 1), activation='sigmoid',name='conv16')(conv6)

        input_weights = Input((256, 256, 1))

        weighted_bcl_loss = partial(weighted_binary_crossentropy_loss_w, weights= input_weights)

        model2 = Model(inputs=[base_model.input, input_weights], outputs=[conv7])

        for layer in model2.layers[:19]:
            layer.trainable = False
        for layer in model2.layers[19:]:
            layer.trainable = True

        self.weighted_bcl_loss = weighted_bcl_loss
        self.model = model2

    def compile(self, optimizer=optimizers.Adam(0.001), metrics=[mean_iou]):
        self.model.compile(optimizer=optimizer, loss=self.weighted_bcl_loss, metrics=metrics)

    def unfreeze_encoder(self,optimizer=optimizers.Adam(0.0003)):
        for layer in self.model.layers:
            layer.trainable = True
        # sgd = optimizers.SGD(lr=0.0003, decay=1e-6, momentum=0.9, nesterov=True)
        self.compile(optimizer)

    def train(self, x, y, epochs=50, batch_size=32, validation_data = None):
        self.model.fit(x, y, batch_size = 32, epochs = epochs, validation_data=validation_data)
