import logging
from os import path

import autokeras as ak
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dense, Dropout, Reshape, \
    Bidirectional, LSTM, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *

import common
from captcha import CaptchaManager
from config import Config
from dl_config import DLTrainConfig
from dl_data_process import DataLoader


def build_CNN_model():
    # Inputs to the model
    input_tensor = Input((DLTrainConfig.width, DLTrainConfig.height, 1), dtype='float32', name='Input')

    # First convolution block
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv_1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, name='MaxPooling_1')(x)
    x = Dropout(0.25, name='Dropout1')(x)

    # Second convolution block
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='Conv_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, name='MaxPooling_2')(x)
    # x = Dropout(0.25, name='Dropout2')(x)

    # Third convolution block
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='Conv_3')(x)
    x = BatchNormalization(name='BN_1')(x)
    # x = Dropout(0.25, name='Dropout3')(x)

    # Fourth convolution block
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='Conv_4')(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=2, name='MaxPooling_3')(x)
    # x = Dropout(0.25, name='Dropout4')(x)

    # Fifth convolution block
    x = Conv2D(512, (2, 2), activation='relu', name='Conv_5')(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=2, name='MaxPooling_4')(x)
    x = Dropout(0.25, name='Dropout5')(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu', name='Dense2')(x)
    # Output layer
    y_pred = Dense(DLTrainConfig.one_hot_length, activation='sigmoid', name='Softmax')(x)

    def loss_fn(y_true, y_pred):
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    # Define the model
    model = Model(inputs=input_tensor, outputs=y_pred, name='CRNN_Model_with_CTC_LOSS')

    # Compile the model and return
    # model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits(), optimizer=Adam())
    model.compile(loss=loss_fn, optimizer=Adam())

    return model


# class MobileV3SmallModel(tf.keras.Model):
#     def __init__(self):
#         super().__init__(name='MobileV3Model')
#         self.mobile_net_v3 = tf.keras.applications.MobileNetV3Small(
#             input_shape=(160, 160, 1),
#             include_top=False)
#         self.pooling = tf.keras.layers.GlobalAveragePooling2D()
#         self.dense = tf.keras.layers.Dense(DLTrainConfig.one_hot_length, activation='softmax')
#
#     def call(self, inputs, training=None, mask=None):
#         x = self.mobile_net_v3(inputs)
#         x = self.pooling(x)
#         return self.dense(x)
#
#     def get_config(self):
#         config = super().get_config().copy()
#         config.update({
#             'mobile_net_v3': self.mobile_net_v3,
#             'pooling': self.pooling,
#             'dense': self.dense,
#         })
#         return config


# class Trainer(object):
#     def __init__(self, model: tf.keras.Model):
#         self.model = model
#         self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#         self.optimizer = tf.keras.optimizers.Adam()
#         self.checkpoint = tf.train.Checkpoint(model=model)
#         self.checkpoint_save_dir = path.join(DLTrainConfig.checkpoint_dir_path, self.model.name)
#
#         self.train_loss = tf.keras.metrics.Mean(name='train_loss')
#         self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
#
#         self.validation_loss = tf.keras.metrics.Mean(name='validation_loss')
#         self.validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='validation_accuracy')
#
#     @tf.function
#     def train_step(self, images, labels):
#         with tf.GradientTape() as tape:
#             # training=True is only needed if there are layers with different
#             # behavior during training versus inference (e.g. Dropout).
#             predictions = self.model(images, training=True)
#             loss = self.loss_object(labels, predictions)
#         gradients = tape.gradient(loss, self.model.trainable_variables)
#         self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
#
#         self.train_loss(loss)
#         self.train_accuracy(labels, predictions)
#
#     @tf.function
#     def validation_step(self, images, labels):
#         # training=False is only needed if there are layers with different
#         # behavior during training versus inference (e.g. Dropout).
#         predictions = self.model(images, training=False)
#         t_loss = self.loss_object(labels, predictions)
#
#         self.validation_loss(t_loss)
#         self.validation_accuracy(labels, predictions)
#
#     def fit(self, train_ds: tf.data.Dataset, validation_ds: tf.data.Dataset):
#         self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_save_dir))
#         for epoch in range(DLTrainConfig.Epoch):
#             # Reset the metrics at the start of the next epoch
#             self.train_loss.reset_states()
#             self.train_accuracy.reset_states()
#             self.validation_loss.reset_states()
#             self.validation_accuracy.reset_states()
#
#             for images, labels in train_ds:
#                 self.train_step(images, labels)
#
#             for test_images, test_labels in validation_ds:
#                 self.validation_step(test_images, test_labels)
#
#             logging.info(f"Epoch {epoch + 1}, " +
#                          f"Loss: {self.train_loss.result()}, " +
#                          f"Accuracy: {self.train_accuracy.result() * 100}, " +
#                          f"Validation Loss: {self.validation_loss.result()}, " +
#                          f"Validation Accuracy: {self.validation_accuracy.result() * 100}")
#             if epoch % 10 == 0:
#                 logging.info(f"saving checkpoint to at epoch {epoch}")
#                 self.checkpoint.save(path.join(self.checkpoint_save_dir, 'model.ckpt'))
#

# def train_mobile_net_v3(mgr: CaptchaManager):
#     def change_range(image):
#         image /= 255
#         return 2 * image - 1
#
#     dl = DataLoader(mgr)
#     train_ds = dl.load_ds(DLTrainConfig.train_dir_path, preprocess_image_func=change_range)
#     validation_ds = dl.load_ds(DLTrainConfig.validation_dir_path, preprocess_image_func=change_range)
#     trainer = Trainer(MobileV3SmallModel())
#     trainer.fit(train_ds=train_ds, validation_ds=validation_ds)


def train_CNN_model(mgr: CaptchaManager):
    model = build_CNN_model()
    model.summary()
    dl = DataLoader(mgr)
    train_ds = dl.load_ds(DLTrainConfig.train_dir_path)
    validation_ds = dl.load_ds(DLTrainConfig.validation_dir_path)
    # Model Check Point
    checkpoint = ModelCheckpoint(path.join(DLTrainConfig.checkpoint_dir_path, 'CRNN', 'CRNN.h5'),  # Filepath
                                 monitor='val_loss',
                                 save_best_only=True,
                                 verbose=1,
                                 mode='auto',
                                 save_weights_only=False,
                                 save_freq='epoch')
    # Add early stopping
    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=DLTrainConfig.min_delta,
                                   patience=DLTrainConfig.early_stopping_patience,
                                   verbose=1,
                                   mode='auto',
                                   baseline=None,
                                   restore_best_weights=True)
    history = model.fit(train_ds, validation_data=validation_ds, epochs=DLTrainConfig.Epoch,
                        callbacks=[checkpoint, early_stopping])
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.subplot()
    plt.plot(range(len(train_loss)), train_loss, label='Train' + "{:10.4f}".format(min(history.history['loss'])))
    plt.plot(range(len(val_loss)), val_loss, label='Valid' + "{:10.4f}".format(min(history.history['val_loss'])))
    plt.legend(loc='upper right')
    plt.title('Loss of CNN')
    plt.show()


def train_auto_keras(mgr: CaptchaManager):
    dl = DataLoader(mgr)
    train_ds = dl.load_ds(DLTrainConfig.train_dir_path)
    validation_ds = dl.load_ds(DLTrainConfig.validation_dir_path)
    test_ds = dl.load_ds(DLTrainConfig.test_dir_path)
    clf = ak.ImageClassifier(overwrite=True)
    # Feed the image classifier with training data.
    clf.fit(
        train_ds, validation_data=validation_ds
    )
    test_loss = clf.evaluate(test_ds)
    logging.info(f'test_loss = {test_loss}')
    preprocess_graph, best_model = clf.tuner.get_best_model()
    best_model.save(path.join(DLTrainConfig.checkpoint_dir_path, 'auto_keras', './best_model.h5'))


if __name__ == '__main__':
    common.init_logger()
    Config.init_from_config_path('./config.json')
    DLTrainConfig.init_from_config(Config.inst())
    m = CaptchaManager.from_captcha_dir(Config.inst().captcha_dir_path)
    train_CNN_model(m)
    # train_auto_keras(m)
