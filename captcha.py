import os
import random
import sys
import time
import logging
import common
import pickle
import re
import shutil
import hashlib

from PIL import Image
from config import Config
from typing import Optional, Dict, Set
from os import path


def hash_md5(s: bytes):
    return hashlib.md5(s).hexdigest()


class CaptchaManager(object):
    TYPE_ORIGINAL = 'original'
    TYPE_LABELED = 'labeled'

    original_image_hashes_filename = 'original_image_hashes.pickle'
    labeled_image_hashes_filename = 'labeled_image_hashes.pickle'

    def __init__(self):
        self.captcha_dir: Optional[str] = None
        self.labeled_dir: Optional[str] = None
        self.max_label_idx: Optional[int] = None
        self.original_dir: Optional[str] = None
        self.rand = random.Random()
        self.labeled_image_filename_regex_pattern = re.compile(r'^([0-9])+_([0-9\-+/:*]{3})_(.*)$')

    @classmethod
    def from_captcha_dir(cls, captcha_dir):
        instance = cls()
        instance.captcha_dir = captcha_dir
        instance.labeled_dir = path.join(instance.captcha_dir, 'labeled')
        instance.original_dir = path.join(instance.captcha_dir, 'original')

        def check_and_mkdir(target_dir: str):
            if path.exists(target_dir):
                assert path.isdir(target_dir), '%s 文件夹被文件占用！' % target_dir
            else:
                os.mkdir(target_dir)

        check_and_mkdir(instance.captcha_dir)
        check_and_mkdir(instance.labeled_dir)
        check_and_mkdir(instance.original_dir)
        max_label = 0
        for f in os.listdir(instance.labeled_dir):
            if not f.endswith('jpeg'):
                continue
            try:
                max_label = max(int(f.split('_')[0]), max_label)
            except ValueError:
                ...
        instance.max_label_idx = max_label
        return instance

    def image_hashes_pickle_filepath(self, pickle_filename) -> str:
        return path.join(self.captcha_dir, pickle_filename)

    @property
    def original_image_hashes_pickle_filepath(self) -> str:
        return self.image_hashes_pickle_filepath(self.original_image_hashes_filename)

    @property
    def labeled_image_hashes_pickle_filepath(self):
        return self.image_hashes_pickle_filepath(self.labeled_image_hashes_filename)

    def save_original(self, content: bytes) -> str:
        name = self.generate_original_image_filename()
        p = path.join(self.original_dir, name)
        self.save_image(content, p)
        return p

    def save_labeled(self, original_image_filepath: str, label_text: str) -> str:
        name = self.generate_labeled_image_filename(path.basename(original_image_filepath), label_text)
        p = path.join(self.labeled_dir, name)
        shutil.copyfile(original_image_filepath, p)
        return p

    @staticmethod
    def save_image(content: bytes, p: str):
        with open(p, 'wb') as f:
            f.write(content)
        logging.info('saved captcha to' + p)

    def generate_original_image_filename(self, ) -> str:
        time_str = time.strftime("%Y_%m_%d-%H_%M_%S_", time.localtime())
        r = self.rand.randint(0, 2 ** 32 - 1)
        return '%s_%d.jpeg' % (time_str, r)

    def decode_labeled_image_filename(self, labeled_image_filename) -> (int, str, str):
        result = self.labeled_image_filename_regex_pattern.match(labeled_image_filename)
        if result:
            return int(result.group(1)), result.group(2), result.group(3)
        return None, None, None

    def generate_labeled_image_filename(self, original_filename, label_text) -> str:
        self.max_label_idx += 1
        return '{}_{}_{}'.format(self.max_label_idx, label_text, original_filename)

    @staticmethod
    def load_image_hashes(pickle_filepath) -> Optional[Dict]:
        if path.exists(pickle_filepath):
            with open(pickle_filepath, 'rb') as f:
                return pickle.load(f)
        return None

    def preprocess_image_hashes(self, image_type: str) -> Dict:
        """
        预处理original与labeled文件夹中的全部图片数据。
        image_hashes = {
            'hash_value': set({'path/to/image1', 'path/to/image2'})
        }
        """
        pickle_filepath_map = {
            self.TYPE_ORIGINAL: self.original_image_hashes_pickle_filepath,
            self.TYPE_LABELED: self.labeled_image_hashes_pickle_filepath
        }
        image_dirs = {
            self.TYPE_ORIGINAL: self.original_dir,
            self.TYPE_LABELED: self.labeled_dir
        }
        pickle_filepath = pickle_filepath_map[image_type]
        image_dir = image_dirs[image_type]
        image_hashes = self.load_image_hashes(pickle_filepath)
        if image_hashes is None:
            image_hashes = {}
        for idx, filename in enumerate(os.listdir(image_dir)):
            if idx % 1000 == 0:
                logging.info('preprocessing pickle {}, idx = {}'.format(pickle_filepath, idx))
            if not filename.endswith('jpeg'):
                continue
            p = path.join(image_dir, filename)
            if not path.exists(p):
                logging.info('file in original path not exists, path = %s' % p)
                continue
            file_size = path.getsize(p)
            if file_size == 0:
                logging.info('file size = 0, delete path = %s' % p)
                os.remove(p)
            if file_size > 5000:
                logging.info('encounter file size > 5000, skip. path = %s' % p)
            with open(p, 'rb') as image:
                hash_value = hash_md5(image.read())
                if hash_value not in image_hashes:
                    image_hashes[hash_value] = set()
                image_hashes[hash_value].add(p)
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(image_hashes, f)

        return image_hashes

    def unlabeled_image_paths(self) -> Set:
        original_hashes = self.load_image_hashes(pickle_filepath=self.original_image_hashes_pickle_filepath)
        labeled_hashes = self.load_image_hashes(pickle_filepath=self.labeled_image_hashes_pickle_filepath)
        if original_hashes is None or labeled_hashes is None:
            assert original_hashes is not None and labeled_hashes is not None, '需要进行预处理'
        labeled_not_contained_hashes = original_hashes.keys() - labeled_hashes.keys()
        logging.info('labeled_not_contained_hashes, size = {}'.format(len(labeled_not_contained_hashes)))
        result_set = set()
        for hash_value in labeled_not_contained_hashes:
            result_set.update(original_hashes[hash_value])

        def labeled_set_contains(labeled_set: Set[str], original_image_filename: str):
            for label_filepath in labeled_set:
                if original_image_filename in label_filepath:
                    return True
            return False
        for hash_value, image_path_set in labeled_hashes.items():
            labeled_image_name_set = {path.basename(p) for p in image_path_set}
            if hash_value not in original_hashes:
                continue
            for original_filepath in original_hashes[hash_value]:
                original_filename = path.basename(original_filepath)
                if labeled_set_contains(labeled_image_name_set, original_filename):
                    continue
                result_set.add(original_filepath)
        for labeled_image_path in self.iter_labeled():
            _, _, original_filename = self.decode_labeled_image_filename(path.basename(labeled_image_path))
            if original_filename is None:
                continue
            original_filepath = path.join(self.original_dir, original_filename)
            result_set.remove(original_filepath)

        logging.info('original image hash_set size = {}'.format(len(original_hashes)))
        logging.info('labeled image hash_set size = {}'.format(len(labeled_hashes)))
        logging.info('unlabeled image paths size = {}'.format(len(result_set)))
        return result_set

    @staticmethod
    def show_image(image_path):
        img = Image.open(image_path)
        img.show()

    def do_label(self):
        unlabeled = self.unlabeled_image_paths()
        for original_image_path in unlabeled:
            self.show_image(original_image_path)
            logging.info('processing original image {}'.format(original_image_path))
            while True:
                label_text = str(input('输入验证码内容：'))
                check = str(input('输入回车确认，输入skip跳过该图片，输入q退出，输入其他字符重试：'))
                if len(label_text) != 3:
                    logging.warning('长度应为3！重试。label_text = {}'.format(label_text))
                    continue
                if not (label_text[0].isdigit() and label_text[1] in ['+', '/', '-', '*'] and label_text[2].isdigit()):
                    logging.warning('不合法！重试。')
                    continue
                if check == '':
                    logging.info('读取到验证码label为：{}。'.format(label_text))
                    if '/' in label_text:
                        label_text = label_text.replace('/', ':')
                        logging.info('替换验证码中的/为:，防止路径与名称混淆，更改后的验证码文本：{}'.format(label_text))
                    labeled_path = self.save_labeled(original_image_path, label_text)
                    logging.info('保存到文件：{}'.format(labeled_path))
                    break
                elif check == 'skip':
                    break
                elif check == 'q':
                    logging.info('退出！')
                    sys.exit(0)

    def iter_original(self):
        for filename in os.listdir(self.original_dir):
            yield path.join(self.original_dir, filename)

    def iter_labeled(self):
        return self.iter_jpeg(self.labeled_dir)
        # for filename in os.listdir(self.labeled_dir):
        #     yield path.join(self.original_dir, filename)

    @staticmethod
    def iter_jpeg(dir_path):
        for filename in os.listdir(dir_path):
            if not filename.endswith('jpeg'):
                continue
            yield path.join(dir_path, filename)


def do_preprocess_image_hashes(m: CaptchaManager, original=False):
    if original:
        m.preprocess_image_hashes(m.TYPE_ORIGINAL)
    m.preprocess_image_hashes(m.TYPE_LABELED)


def do_label(m: CaptchaManager):
    m.do_label()


def do_show_image_hashes_example(m: CaptchaManager):
    image_hashes = m.load_image_hashes(m.original_image_hashes_pickle_filepath)
    if image_hashes is None:
        logging.info('需要预处理original image hashes')
        return
    for hash_value, s in image_hashes.items():
        if len(s) > 1:
            logging.info('hash_value = {}, set = {}'.format(hash_value, s))


if __name__ == '__main__':
    common.init_logger()
    Config.init_from_config_path('./config.json')
    mgr = CaptchaManager.from_captcha_dir(Config.inst().captcha_dir_path)
    do_preprocess_image_hashes(mgr)
    # do_show_image_hashes_example(mgr)
    do_label(mgr)
