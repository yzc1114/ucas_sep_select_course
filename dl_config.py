from os import path

from config import Config


class DLTrainConfig(object):
    width = 160  # 验证码图片的宽
    height = 60  # 验证码图片的高
    one_hot_length = 24
    data_segmentation = (0.8, 0.1, 0.1)  # 训练集，验证集和测试集数量
    op_map = {
        '+': 0,
        '-': 1,
        '*': 2,
        '/': 3,
        ':': 3
    }
    op_map_inverted = [{v: k} for k, v in op_map.items()]
    early_stopping_patience = 10
    min_delta = 0.0001

    test_dir_path = None
    train_dir_path = None
    validation_dir_path = None
    tensorboard_dir_path = None  # tensorboard的log路径
    checkpoint_dir_path = None

    alpha = 1e-3  # 学习率
    Epoch = 100  # 训练轮次
    batch_size = 32  # 批次数量
    keep_prob = 0.5  # dropout比例
    print_per_batch = 20  # 每多少次输出结果
    save_per_batch = 20

    @classmethod
    def init_from_config(cls, config: Config):
        cls.test_dir_path = path.join(config.captcha_dir_path, 'test')
        cls.train_dir_path = path.join(config.captcha_dir_path, 'train')
        cls.validation_dir_path = path.join(config.captcha_dir_path, 'validation')
        cls.tensorboard_dir_path = path.join(config.captcha_dir_path, 'tensorboard')
        cls.checkpoint_dir_path = path.join(config.captcha_dir_path, 'checkpoints')
