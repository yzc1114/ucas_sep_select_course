import logging
from os import path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import common
from captcha import CaptchaManager
from config import Config
from dl_config import DLTrainConfig
from dl_data_process import DataLoader

common.init_logger()
Config.init_from_config_path('./config.json')
DLTrainConfig.init_from_config(Config.inst())


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def build_crnn_model():
    # Inputs to the model
    input_img = layers.Input(
        shape=(DLTrainConfig.width, DLTrainConfig.height, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((DLTrainConfig.width // 4), (DLTrainConfig.height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)
    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(
        len(DLTrainConfig.characters) + 1, activation="softmax", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    return model


def train_model(model: tf.keras.Model, mgr: CaptchaManager):
    model.summary()
    dl = DataLoader(mgr)
    train_ds = dl.load_ds(DLTrainConfig.train_dir_path)
    validation_ds = dl.load_ds(DLTrainConfig.validation_dir_path)
    # Model Check Point
    checkpoint = ModelCheckpoint(path.join(DLTrainConfig.checkpoint_dir_path, model.name, f'{model.name}.h5'),
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
    logging.info(f'train_loss:{train_loss}, val_loss: {val_loss}')


if __name__ == '__main__':
    m = CaptchaManager.from_captcha_dir(Config.inst().captcha_dir_path)
    train_model(build_crnn_model(), m)
