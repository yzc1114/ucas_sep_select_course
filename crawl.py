import logging
import common

from jwxk_base import JWXK_Base
from captcha import CaptchaManager
from config import Config
from retrying import retry
from requests_html import HTMLSession


class CaptchaCrawler(JWXK_Base):
    def __init__(self, captcha_manager: CaptchaManager):
        super().__init__(captcha_manager)

    def crawl(self):
        session = HTMLSession()
        self._auth(session)

        @retry(
            stop_max_attempt_number=1000000,
            retry_on_exception=common.retry_with_log
        )
        def crawl_image():
            response = session.get("http://jwxk.ucas.ac.cn/captchaImage", timeout=Config.inst().timeout)
            if len(response.content) > 0:
                self._save_captcha(response.content)
            raise common.LoopRetryError()

        crawl_image()


if __name__ == '__main__':
    common.init_logger()
    Config.init_from_config_path('./config.json')
    captcha_mgr = CaptchaManager.from_captcha_dir('./captcha')
    crawler = CaptchaCrawler(captcha_mgr)
    crawler.crawl()
