import json
from typing import Dict, List, Optional

from model import Department, Course


class Config(object):
    instance: Optional['Config'] = None

    def __init__(self):
        self.inited = False
        self.dep2courses: Dict[Department, List[Course]] = {}
        self.captcha_dir_path: Optional[str] = None
        self.timeout: int = 5
        self.best_weights_path: Optional[str] = None
        self.retry_config: Dict = {
            'max_attempts': 10000,
            'interval_seconds': 2
        }

    @classmethod
    def init_from_config_path(cls, config_path):
        if cls.instance is not None:
            return
        with open(config_path) as f:
            conf_json = json.load(f)
            config = cls()
            config.captcha_dir_path = conf_json.get('captcha_dir_path', None)
            config.retry_config = {}
            config.retry_config = dict(conf_json.get('retry', config.retry_config))
            config.best_weights_path = conf_json['best_weights_path']
            config_courses = conf_json['courses']
            for dep_name, courses in config_courses.items():
                dep = Department(dep_name)
                res_courses: List[Course] = []
                for course in courses:
                    c = Course(course['course_name'], course['course_code'])
                    c.department = dep
                    res_courses.append(c)
                config.dep2courses[dep] = res_courses
            cls.instance = config

    @classmethod
    def inst(cls) -> 'Config':
        return cls.instance
