import logging
import os
import sys
import shutil
import tensorflow as tf
import numpy as np
import common
import matplotlib.pyplot as plt

from captcha import CaptchaManager
from config import Config
from dl_config import DLTrainConfig
from os import path
from PIL import Image


def label2vec(label: str) -> tf.Tensor:
    """"
    将验证码标签转为24维的向量，后置操作符。
    :param label: 3*7
    :return:
    [0,0,0,1,0,0,0,0,0,0,
     0,0,0,0,0,0,0,7,0,0,
     0,0,1,0] // 按照+ - * /的顺序
     """
    label_vec = np.zeros(24)
    left_num_idx = int(label[0])
    right_num_idx = 10 + int(label[2])
    op_idx = 20 + DLTrainConfig.op_map[label[1]]
    label_vec[left_num_idx] = 1
    label_vec[right_num_idx] = 1
    label_vec[op_idx] = 1
    return tf.convert_to_tensor(label_vec)


def vec2label(vec: tf.Tensor) -> str:
    label = ''
    for idx in np.nonzero(vec.numpy()):
        if idx < 20:
            label += str(idx % 10)
        else:
            label += str(DLTrainConfig.op_map_inverted[idx - 20])
    return label


class Generate:
    def __init__(self, mgr: CaptchaManager):
        self.check_path(DLTrainConfig.test_dir_path)
        self.check_path(DLTrainConfig.validation_dir_path)
        self.check_path(DLTrainConfig.train_dir_path)
        self.captcha_mgr: CaptchaManager = mgr
        self.run()

    @staticmethod
    def check_path(dir_path):
        # 检查文件夹是否存在，不存在就创建
        if os.path.exists(dir_path):
            logging.error('生成数据集时，需保证目标的文件夹不存在，路径为：{}'.format(dir_path))
            sys.exit(-1)
        os.mkdir(dir_path)

    def run(self):
        logging.info('==> 正在从labeled文件夹中生成训练、验证、测试数据集，比例为({})'.format(DLTrainConfig.data_segmentation))
        image_paths = [p for p in self.captcha_mgr.iter_labeled()]
        total_count = len(image_paths)
        curr_count = 0.
        train_seg, validation_seg, test_seg = DLTrainConfig.data_segmentation

        for idx, image_path in enumerate(image_paths):
            image_filename = path.basename(image_path)
            _, label, original_filename = self.captcha_mgr.decode_labeled_image_filename(image_filename)
            if label is None or original_filename is None:
                continue
            curr_count += 1
            ratio = curr_count / total_count
            if ratio < train_seg:
                shutil.copyfile(image_path, path.join(DLTrainConfig.train_dir_path, image_filename))
            elif ratio < train_seg + validation_seg:
                shutil.copyfile(image_path, path.join(DLTrainConfig.validation_dir_path, image_filename))
            elif ratio < train_seg + validation_seg + test_seg:
                shutil.copyfile(image_path, path.join(DLTrainConfig.test_dir_path, image_filename))


class DataLoader:
    def __init__(self, mgr: CaptchaManager):
        self.captcha_mgr: CaptchaManager = mgr
        self.test_images = os.listdir(DLTrainConfig.test_dir_path)
        self.train_images = os.listdir(DLTrainConfig.train_dir_path)
        self.sample_num = len(self.train_images)

    def read_data(self, p: str):
        img = Image.open(p).convert('L')
        image_array = np.array(img)
        image_data = image_array.flatten() / 255.0
        # 切割图片路径
        image_name = path.basename(p)
        _, label, _ = self.captcha_mgr.decode_labeled_image_filename(image_name)
        label_vec = label2vec(label)
        return image_data, label_vec

    @classmethod
    def preprocess_image(cls, image, preprocess_func=None):
        image = tf.image.decode_jpeg(image, channels=1)
        # image = tf.expand_dims(image[:, :, 1], -1)
        # print(image.shape)
        tf.image.resize(image, [DLTrainConfig.height, DLTrainConfig.width])
        image = tf.where(image > 50, 255, 0)
        image = tf.transpose(image, perm=[1, 0, 2])
        if preprocess_func is None:
            image /= 255  # normalize to [0,1] range
        else:
            image = preprocess_func(image)
        return image

    @classmethod
    def load_and_preprocess_image(cls, image_path, preprocess_image_func: None):
        image = tf.io.read_file(image_path)
        return cls.preprocess_image(image, preprocess_image_func)

    def parse_label_vec(self, image_path) -> tf.Tensor:
        return label2vec(self.parse_label_text(image_path))

    def parse_label_text(self, image_path) -> str:
        image_name = path.basename(image_path)
        _, label, _ = self.captcha_mgr.decode_labeled_image_filename(image_name)
        return label

    def load_ds(self, images_dir_path, preprocess_image_func=None):
        image_paths = [path.join(images_dir_path, p) for p in os.listdir(images_dir_path)]
        labels = list(map(lambda p: self.parse_label_vec(p), image_paths))
        # label_texts = list(map(lambda p: self.parse_label_text(p), image_paths))
        # path_ds = tf.data.Dataset.from_tensor_slices(image_paths)

        def load_and_preprocess_image(image):
            return self.load_and_preprocess_image(image, preprocess_image_func=preprocess_image_func)
        images = list(map(lambda p: load_and_preprocess_image(p), image_paths))
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        ds = ds.shuffle(buffer_size=len(image_paths))
        ds = ds.batch(DLTrainConfig.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds


def test_dataloader(mgr: CaptchaManager):
    dl = DataLoader(mgr)

    def change_range(image):
        image /= 255
        return 2*image-1
    image_label_ds = dl.load_ds(DLTrainConfig.train_dir_path)
    validation = dl.load_ds(DLTrainConfig.validation_dir_path)
    # for image_path, label_text, image, label in image_label_ds.take(4):
    #     print(image_path, label_text, image, label)
    for images, label in image_label_ds.take(1):
        print(images[0], label)
        plt.imshow(images[0])
        plt.grid(False)
        plt.show()


def generate_training_data(mgr: CaptchaManager):
    Generate(mgr)


if __name__ == '__main__':
    common.init_logger()
    Config.init_from_config_path('./config.json')
    DLTrainConfig.init_from_config(Config.inst())
    m = CaptchaManager.from_captcha_dir(Config.inst().captcha_dir_path)
    test_dataloader(m)
