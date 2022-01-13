import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

import common
from captcha import CaptchaManager
from config import Config
from dl_config import DLTrainConfig
from dl_data_process import ints_to_label_chars, DataLoader
from dl_train import build_crnn_model

common.init_logger()
Config.init_from_config_path('./config.json')
DLTrainConfig.init_from_config(Config.inst())


def build_predict_model():
    model = build_crnn_model()
    model.load_weights(Config.inst().best_weights_path)
    model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    model.summary()
    return model


# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
              :, :DLTrainConfig.max_length]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(ints_to_label_chars(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


class Predictor(object):
    def __init__(self, dl: DataLoader):
        self.data_loader = dl
        self.model = build_predict_model()

    def predict_one(self, image_path) -> str:
        image = self.data_loader.load_and_preprocess_image(image_path)
        images = tf.expand_dims(image, 0)
        preds = self.model.predict(images)
        pred_text = decode_batch_predictions(preds)[0]
        logging.info(f'识别图片{image_path}的结果为{pred_text}')
        return pred_text


def predict_on_validation(dl: DataLoader):
    validation_ds = dl.load_ds(DLTrainConfig.validation_dir_path)
    model = build_predict_model()
    #  Let's check results on some validation samples
    for batch in validation_ds.take(1):
        batch_images = batch["image"]
        batch_labels = batch["label"]
        preds = model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds)

        orig_texts = []
        for label in batch_labels:
            label = tf.strings.reduce_join(ints_to_label_chars(label)).numpy().decode("utf-8")
            orig_texts.append(label)

        _, ax = plt.subplots(4, 4, figsize=(15, 5))
        for i in range(len(pred_texts)):
            img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
            img = img.T
            title = f"Prediction: {pred_texts[i]}"
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")
        plt.show()


if __name__ == '__main__':
    m = CaptchaManager.from_captcha_dir(Config.inst().captcha_dir_path)
    data_loader = DataLoader(m)
    predict_on_validation(data_loader)
