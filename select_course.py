#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import re
from typing import List, Tuple, Optional

from requests_html import HTML
from requests_html import HTMLSession
from retrying import retry

import common
from captcha import CaptchaManager
from config import Config
from dl_data_process import DataLoader
from dl_predict import Predictor
from sep_base import SepBase

common.init_logger()
Config.init_from_config_path('./config.json')


class CourseSelector(SepBase):
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
        self.predictor: Predictor = Predictor(DataLoader(captcha_manager))
        self.predict_result_regex = re.compile(r'^[0-9][+\-*:][0-9]=\?$')

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
                elem = response.html.xpath(f"//span[contains(., '{course.course_code}')]",
                                           first=True)
                if elem is None:
                    logging.info(f'未在选课页面查找到课程：{course.course_name}, {course.course_code}，课程可能已经选上，或者配置的课程号与学院无法对应。')
                    continue
                course_id_raw = elem.attrs["id"]  # 元素id为：'courseCode_xxxx' 虽然写着courseCode，但是我叫它id。
                course.course_id = course_id_raw.split('_')[-1]

        captcha = response.html.xpath('//*[@id="adminValidateImg"]', first=True)
        captcha_src = "http://jwxk.ucas.ac.cn" + captcha.attrs['src']
        response = session.get(captcha_src, timeout=Config.inst().timeout)
        captcha_path = self._save_captcha(response.content)
        # 手动
        # im = plt.imread(captcha_path)
        # plt.imshow(im)
        # plt.show()
        # v_code = str(input('输入验证码：'))

        # 自动识别验证码
        v_code = self.predict_captcha(captcha_path)
        if v_code is None:
            raise common.LoopRetryError()

        return url_param, csrf_token, v_code

    def predict_captcha(self, captcha_path) -> Optional[str]:
        pred_result = self.predictor.predict_one(captcha_path)
        match_result = self.predict_result_regex.match(pred_result)
        if match_result:
            pred_result = pred_result.replace(':', '/')
            pred_result = pred_result[:3]
            try:
                return str(int(eval(pred_result)))
            except SyntaxError as e:
                logging.error(f'使用深度学习模型识别的结果在eval时报错，预测结果为：{pred_result}，报错为：{e}')
                return None
        else:
            logging.warning(f'使用深度学习模型识别的结果不合法，结果为：{pred_result}')
            return None

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

        response_html_doc = response.content.decode('utf-8')
        html = HTML(html=response_html_doc)
        results = html.search_all("课程[{}]选课成功")
        if results:
            for result in results:
                logging.info(f"选课成功：{result.fixed}")
        msg = response.html.xpath('//*[@id="loginError"]', first=True)
        if msg is not None:
            logging.info(msg.text)
        raise common.LoopRetryError()

    @retry(
        stop_max_attempt_number=Config.inst().retry_config['max_attempts'],
        wait_random_min=1000 * Config.inst().retry_config['interval_seconds_range'][0],
        wait_random_max=1000 * Config.inst().retry_config['interval_seconds_range'][1],
        retry_on_exception=common.retry_with_log
    )
    def select_courses(self):
        session = HTMLSession()
        self._auth(session)
        url_param, csrf, v_code = self.__extract_essential_params(session)
        self.__save_courses(session, url_param, csrf, v_code)


def main():
    captcha_manager = CaptchaManager.from_captcha_dir('./captcha')
    cs = CourseSelector(captcha_manager)
    cs.select_courses()


if __name__ == "__main__":
    main()
