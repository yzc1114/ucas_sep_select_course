# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
#
# import logging
# import sys
# import time
#
# import browsercookie
# from requests_html import HTMLSession
# from retrying import retry
#
#
# def retry_if_failed(exception):
#     logging.debug(exception)
#     return isinstance(exception, Exception)
#
#
# @retry(
#     # stop_max_attempt_number=3,
#     wait_fixed=5000,
#     retry_on_exception=retry_if_failed,
# )
# def main():
#     logging.info("main")
#     cookies = browsercookie.chrome()
#
#     if not cookies:
#         logging.error("Cookie 获取失败，请检查")
#         return
#
#     timeout = 5
#     session = HTMLSession()
#
#     # 马克思主义学院	010108MGB001H-08	自然辩证法概论
#
#     # 人工智能学院	081104M05012H	知识图谱
#
#     # 人工智能学院	081104M05017H	脑认知机理与计算模型
#
#     # 体育教研室	045200MGX001H-01	男子健身
#     # 心理学系	040200MGX004H	生活中的心理学
#     # 经济与管理学院	120100MGX042H	生活中的经济学
#     # 外语系	050200MGX003H-01	学术交流英语口语-01班
#
#     # 1.19号：
#
#     # 外语系	050200DGB001H-S273	英语B-273班（怀）-高级口语
#     # 外语系	050200DGB001H-S58	英语B-58班（怀）-高级口语
#
#     # 外语系	050200DGB001H-W62	英语B-62班（怀）-学术论文写作（计算机科学）
#     # 外语系	050200DGB001H-W302	英语B-302班（怀）-学术论文写作（综合）
#
#     # 体育教研室 045200MGX001H-01 男子健身
#     # 体育教研室 045200MGX011H-03 男子篮球普修
#     # 体育教研室 045200MGX011H-04 男子篮球普修
#
#     # 心理学系 040200MGX004H 生活中的心理学
#     # 经管学院 120100MGX042H 生活中的经济学
#     # 外语系 050200MGX003H-01 学术交流英语口语-01班
#
#
#     departments = [
#         # "体育教研室",
#         # "人工智能学院",
#         "计算机学院",
#         # "马克思主义学院",
#         # "公管学院",
#         # "心理学系",
#         # "经济与管理学院",
#         # "经管学院",
#         # "外语系"
#     ]
#     course_codes = [  # 课程编码
#         "081202M04003H"        # 算法
#         # "050200MGB003H-201", # 英语慕课
#         # "081201M04002H", # 计算机体系结构
#         # "081202M05010H", # 并行
#         # "081202M05005H", # 编译原理
#         # "010108MGB001H-08", # 自然辩证法概论
#         # "081104M05012H",    # 知识图谱
#         # "081104M05017H",    # 脑认知机理与计算模型
#         # "045200MGX010H-06", # 羽毛球
#         # "045200MGX001H-01", # 男子健身
#         # "045200MGX011H-03", # 男子篮球普修
#         # "045200MGX011H-03", # 男子篮球普修
#         # "040200MGX004H",    # 生活中的心理学
#         # "120100MGX042H",    # 生活中的经济学
#         # "050200MGX003H-01", # 学术交流英语口语-01班
#     ]
#     course_ids = []  # 课程 ID
#     # course_ids = ['511076AC9E917DC9', '454B30E6D0635529', '1C2E2D295D7DA5C6', '52E18F2CB232649B', '73579E1454D65B36', '590BD1AEB25643D3', '762AF29E1E9C5556']
#
#     if not course_ids and not (course_codes or departments):
#         logging.error("请设置课程编码与对应学院或课程 id")
#         sys.exit(0)
#
#     while True:
#         response = session.get(
#             "http://sep.ucas.ac.cn/portal/site/226/821",
#             cookies=cookies,
#             timeout=timeout,
#         )
#         # print(response.html.html)
#         url = response.html.xpath(
#             '//*[@id="main-content"]/div/div[2]/div/h4/a', first=True
#         ).attrs["href"]
#         response = session.get(url, timeout=timeout)
#
#         response = session.get(
#             "http://jwxk.ucas.ac.cn/courseManage/main", timeout=timeout
#         )
#         param = (
#             response.html.xpath('//*[@id="regfrm2"]', first=True)
#                 .attrs["action"]
#                 .split("?")[1]
#         )
#
#         # if not course_ids:
#         if True:
#             dept_ids = []
#             for dept in departments:
#                 elem = response.html.xpath(
#                     "//*[@id=\"regfrm2\"]/div/div/label[. = '%s']" % dept, first=True
#                 )
#                 if elem:
#                     dept_ids.append(elem.attrs["for"].split("_")[1])
#             logging.info(f"dept_ids = {str(dept_ids)}")
#
#             response = session.post(
#                 "http://jwxk.ucas.ac.cn/courseManage/selectCourse?" + param,
#                 data={"deptIds": dept_ids, "sb": 0},
#                 timeout=timeout,
#                 )
#
#             _csrftoken_elem = response.html.xpath('//*[@id="_csrftoken"]', first=True)
#             _csrftoken = None
#             if _csrftoken_elem is not None:
#                 _csrftoken = _csrftoken_elem.attrs["value"]
#
#             course_ids = []
#             for course_code in course_codes:
#                 elem = response.html.xpath(
#                     # '//*[@id="regfrm"]/table/tbody/tr[td[3][. = "%s"]]/td[1]/input'
#                     '//*[@id="regfrm"]/table/tbody/tr[td[4]/a/span[. = "%s"]]/td[1]/input'
#                     % course_code,
#                     first=True,
#                     )
#                 if elem:
#                     course_ids.append(elem.attrs["value"])
#
#             # print("course_ids = ", course_ids, sep="")
#             logging.info(f"course_ids = {str(course_ids)}")
#
#         # course_ids = ["2F324B95F1BBA1E2"]
#         flag = False
#         while True:
#             try:
#                 # _csrftoken=b1229246-e24d-4e48-b86b-391b0fe5020f&deptIds=951&deptIds=927&sids=758B248403AE3ABE&sids=483E08028C4BA52F&vcode=8
#                 response = session.post(
#                     "http://jwxk.ucas.ac.cn/courseManage/saveCourse?" + param,
#                     data={"deptIds": dept_ids, "sids": course, "_csrftoken": _csrftoken, "vcode": },
#                     timeout=timeout,
#                     headers={"Referer": "http://jwxk.ucas.ac.cn/courseManage/selectCourse?" + param}
#                 )
#             except Exception:
#                 logging.info("timeout")
#                 flag = True
#                 break
#
#             error = response.html.search("你的会话已失效或身份已改变，请重新登录")
#             if error:
#                 logging.info("失效")
#                 flag = True
#                 break
#
#             result = response.html.search("课程[{}]选课成功")
#             if result:
#                 logging.info(result[0] + "选课成功！！！")
#             else:
#                 msg = response.html.xpath('//*[@id="loginError"]', first=True)
#                 logging.info("选课失败: " + "None" if msg is None else msg.text)
#                 raise Exception()
#             sys.stdout.flush()
#             time.sleep(1)
#             for course in course_ids:
#                 if flag:
#                     break
#
#
#             # time.sleep(1)
#             if flag:
#                 break
#
#
# if __name__ == "__main__":
#     logging.basicConfig(
#         format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s", level=logging.INFO
#     )
#     main()
#     logging.error("请重新在 Chrome 中登陆 sep")
