import logging
import os
import shutil
import sys
from os import path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import common
from captcha import CaptchaManager
from config import Config
from dl_config import DLTrainConfig


def label2int_tensor(label: str) -> tf.Tensor:
    chars = DLTrainConfig.characters
    st_list = []
    label = label + '=?'
    for s in label:
        s_index = chars.index(s)
        st_list.append(s_index)
    return tf.convert_to_tensor(st_list, dtype=tf.float32)


def ints_to_label_chars(indices: tf.Tensor) -> tf.Tensor:
    l = []
    for index in indices:
        if index < len(DLTrainConfig.characters):
            l.append(DLTrainConfig.characters[int(index.numpy())])
        else:
            l.append(DLTrainConfig.characters[-1])  # '[UNK]'
    return tf.convert_to_tensor(l, dtype=tf.string)


def label2vec(label: str) -> tf.Tensor:
    """"
    将验证码标签转为5 * 17维的向量，
    :param label: 3*7=?
    :return:
    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0
     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
     0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0
     0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0  // 固定为=
     0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0] // 固定为?
     """
    label_vec = np.zeros(5 * len(DLTrainConfig.characters))
    for idx, c in enumerate(label):
        label_vec[len(DLTrainConfig.characters) * idx + DLTrainConfig.characters.index(c)] = 1
    label_vec[3 * len(DLTrainConfig.characters) + DLTrainConfig.characters.index('=')] = 1
    label_vec[4 * len(DLTrainConfig.characters) + DLTrainConfig.characters.index('?')] = 1
    return tf.convert_to_tensor(label_vec)


def vec2label(vec: tf.Tensor) -> str:
    label = ''
    for idx in np.nonzero(vec.numpy()):
        label += str(DLTrainConfig.characters[idx % len(DLTrainConfig.characters)])
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
        self.char_to_num = None
        self.num_to_char = None

    @classmethod
    def preprocess_image(cls, image):
        image = tf.image.decode_jpeg(image, channels=1)
        tf.image.resize(image, [DLTrainConfig.height, DLTrainConfig.width])
        image = tf.where(image > 50, 255, 0)
        image = tf.transpose(image, perm=[1, 0, 2])
        image /= 255
        return image

    @classmethod
    def load_and_preprocess_image(cls, image_path):
        image = tf.io.read_file(image_path)
        return cls.preprocess_image(image)

    def parse_label_vec(self, image_path) -> tf.Tensor:
        return label2vec(self.parse_label_text(image_path))

    def parse_label_int_tensor(self, image_path) -> tf.Tensor:
        return label2int_tensor(self.parse_label_text(image_path))

    def parse_label_text(self, image_path) -> str:
        image_name = path.basename(image_path)
        _, label, _ = self.captcha_mgr.decode_labeled_image_filename(image_name)
        return label

    def load_ds(self, images_dir_path):
        image_paths = [path.join(images_dir_path, p) for p in os.listdir(images_dir_path)]
        labels = list(map(lambda p: self.parse_label_int_tensor(p), image_paths))

        images = list(map(lambda p: self.load_and_preprocess_image(p), image_paths))
        images_labels_ds = tf.data.Dataset.from_tensor_slices((images, labels))

        def combine(image, label):
            return {'image': image, 'label': label}

        ds = images_labels_ds.map(combine)
        ds = ds.shuffle(buffer_size=len(image_paths))
        ds = ds.batch(DLTrainConfig.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds


def test_dataloader(mgr: CaptchaManager):
    dl = DataLoader(mgr)

    image_label_ds = dl.load_ds(DLTrainConfig.train_dir_path)
    for image_label_dict in image_label_ds.take(1):
        print(image_label_dict['image'][0], image_label_dict['label'][0])
        plt.imshow(image_label_dict['image'][0])
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
