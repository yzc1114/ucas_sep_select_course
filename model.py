from typing import Optional


class Department(object):
    def __init__(self, department_name):
        self.department_name: str = department_name
        self.department_id: Optional[int] = None


class Course(object):
    def __init__(self, course_name: str, course_code: str):
        self.course_name: str = course_name
        self.course_code: str = course_code
        self.department: Optional[Department] = None
        self.course_id: Optional[str] = None
