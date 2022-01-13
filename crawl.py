from requests_html import HTMLSession
from retrying import retry

import common
from captcha import CaptchaManager
from config import Config
from sep_base import SepBase

common.init_logger()
Config.init_from_config_path('./config.json')


class CaptchaCrawler(SepBase):
    def __init__(self, captcha_manager: CaptchaManager):
        super().__init__(captcha_manager)

    def crawl(self):
        session = HTMLSession()
        self._auth(session)

        @retry(
            stop_max_attempt_number=Config.inst().retry_config['max_attempt'],
            retry_on_exception=common.retry_with_log
        )
        def crawl_image():
            response = session.get("http://jwxk.ucas.ac.cn/captchaImage", timeout=Config.inst().timeout)
            if len(response.content) > 0:
                self._save_captcha(response.content)
            raise common.LoopRetryError()

        crawl_image()


if __name__ == '__main__':
    captcha_mgr = CaptchaManager.from_captcha_dir('./captcha')
    crawler = CaptchaCrawler(captcha_mgr)
    crawler.crawl()
