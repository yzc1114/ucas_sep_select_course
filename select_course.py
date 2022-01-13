#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import common
from requests_html import HTMLSession
from retrying import retry
from config import Config
from jwxk_base import JWXK_Base
from captcha import CaptchaManager
from typing import List, Tuple
import matplotlib.pyplot as plt


class CourseSelector(JWXK_Base):
    """
        config: dict 选课配置。
            示例：
                {
                    "courses": {
                        "计算机学院": [
                            {
                                "name": "算法",
                                "course_code": "081202M04003H"
                            }
                        ]
                    }
                }
    """

    def __init__(self, captcha_manager: CaptchaManager):
        super().__init__(captcha_manager)

    def __extract_essential_params(self, session: HTMLSession) -> Tuple[str, str, str]:
        # 访问选课页面的主页，获取后续请求中需要带上的query参数。
        response = session.get(
            "http://jwxk.ucas.ac.cn/courseManage/main", timeout=Config.inst().timeout
        )
        url_param = response.html.xpath('//*[@id="regfrm2"]', first=True).attrs["action"].split("?")[1]

        dep2courses = Config.inst().dep2courses
        for dep in dep2courses:
            elem = response.html.xpath("//*[@id=\"regfrm2\"]/div/div/label[. = '%s']" % dep.department_name, first=True)
            dep.department_id = int(elem.attrs["for"].split("_")[1]) if elem else None

        dept_ids = [dep.department_id for dep in dep2courses.keys() if dep.department_id is not None]

        response = session.post(
            "http://jwxk.ucas.ac.cn/courseManage/selectCourse?" + url_param,
            data={"deptIds": dept_ids, "sb": 0},
            timeout=Config.inst().timeout)

        csrf_token_elem = response.html.xpath('//*[@id="_csrftoken"]', first=True)
        csrf_token = csrf_token_elem.attrs["value"] if csrf_token_elem is not None else None

        for dep, courses in dep2courses.items():
            for course in courses:
                elem = response.html.xpath('//*[@id="regfrm"]/table/tbody/tr[td[4]/a/span[. = "%s"]]/td[1]/input'
                                           % course.course_code, first=True)
                course.course_id = elem.attrs["value"] if elem else None

        captcha = response.html.xpath('//*[@id="adminValidateImg"]', first=True)
        captcha_src = "http://jwxk.ucas.ac.cn" + captcha.attrs['src']
        response = session.get(captcha_src, timeout=Config.inst().timeout)
        captcha_path = self._save_captcha(response.content)
        im = plt.imread(captcha_path)
        plt.imshow(im)
        plt.show()

        v_code = str(input('输入验证码：'))

        # TODO 识别验证码

        return url_param, csrf_token, v_code

    def __make_save_courses_body(self, csrf: str, v_code: str) -> List[Tuple]:
        """
        生成一个saveCourse接口的body结构。返回Tuple的List，从而让session的post使用form-urlencoded格式转换数据。
        :param csrf:
        :param v_code:
        :return:
        """
        result: List[Tuple] = []
        dept_ids: List[Tuple] = []
        course_ids: List[Tuple] = []
        for dep, courses in Config.inst().dep2courses.items():
            if dep.department_id is None:
                continue
            dept_ids.append(('deptIds', dep.department_id))
            course_ids += [('sids', c.course_id) for c in courses if c.course_id is not None]

        result.append(('_csrftoken', csrf))
        result.extend(dept_ids)
        result.extend(course_ids)
        result.append(('vcode', v_code))
        return result

    def __save_courses(self, session: HTMLSession, url_param: str, csrf: str, v_code: str):
        data = self.__make_save_courses_body(csrf, v_code=v_code)
        response = session.post(
            "http://jwxk.ucas.ac.cn/courseManage/saveCourse?" + url_param,
            data=data,
            timeout=Config.inst().timeout,
            headers={"Referer": "http://jwxk.ucas.ac.cn/courseManage/selectCourse?" + url_param}
        )
        error = response.html.search("你的会话已失效或身份已改变，请重新登录")
        if error:
            logging.error("会话失效")
            return

        result = response.html.search("课程[{}]选课成功")
        if result:
            logging.info("选课成功：" + result)
        msg = response.html.xpath('//*[@id="loginError"]', first=True)
        if msg is not None:
            logging.info("选课失败: " + msg.text)
        raise common.LoopRetryError()

    @retry(
        wait_fixed=1000,
        retry_on_exception=common.retry_with_log
    )
    def select_courses(self):
        session = HTMLSession()
        self._auth(session)
        url_param, csrf, v_code = self.__extract_essential_params(session)
        self.__save_courses(session, url_param, csrf, v_code)


def main():
    common.init_logger()
    captcha_manager = CaptchaManager.from_captcha_dir('./captcha')
    cs = CourseSelector(captcha_manager)
    cs.select_courses()


if __name__ == "__main__":
    main()
