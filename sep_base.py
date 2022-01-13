from typing import Optional

import browsercookie
from requests_html import HTMLSession

from captcha import CaptchaManager
from config import Config


class SepBase(object):
    def __init__(self, captcha_manager: CaptchaManager):
        self.captcha_mgr: CaptchaManager = captcha_manager

    def _auth(self, session: HTMLSession):
        cookies = browsercookie.chrome()
        assert cookies, "Cookie获取失败，请查看是否安装了chrome浏览器"

        # 获取跳转选课网站的链接。
        response = session.get(
            "http://sep.ucas.ac.cn/portal/site/226/821",
            cookies=cookies,
            timeout=Config.inst().timeout,
        )
        url = response.html.xpath(
            '//*[@id="main-content"]/div/div[2]/div/h4/a', first=True
        ).attrs["href"]

        # 访问选课网站的login，使当前session具有合适的上下文信息。
        _ = session.get(url, timeout=Config.inst().timeout)

    def _save_captcha(self, content: bytes) -> Optional[str]:
        return self.captcha_mgr.save_original(content)
