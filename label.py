from enum import Enum


class PHI(Enum):
    No_PHI = 0
    Name = 1
    Address = 2
    Date = 3
    Phone = 4
    Fax = 5
    Email = 6
    Image = 7
    SSN = 8  # 주민등록번호
    Medical = 9  # 진료기록번호
    HealthPlan = 10  # 건강보험번호
    Account = 11  # 계좌번호
    License = 12  # 면허번호
    Vehicle = 13  # 차량식별번호
    Device = 14  # 장치식별번호
    URL = 15
    IP = 16
    Biometric = 17  # 지문 등등
    Other = 18


i2b2_2014 = {
            'DATE': PHI.Date,
            'DOCTOR': PHI.Name,
            'HOSPITAL': PHI.Address,
            'PATIENT': PHI.Name,
            'AGE': PHI.Date,
            'MEDICALRECORD': PHI.Medical,
            'CITY': PHI.Address,
            'STATE': PHI.Address,
            'PHONE': PHI.Phone,
            'USERNAME': PHI.Other,
            'IDNUM': PHI.Other,
            'PROFESSION': PHI.Other,
            'STREET': PHI.Address,
            'ZIP': PHI.Address,
            'ORGANIZATION': PHI.Other,
            'COUNTRY': PHI.Address,
            'FAX': PHI.Fax,
            'DEVICE': PHI.Device,
            'EMAIL': PHI.Email,
            'LOCATION-OTHER': PHI.Other,
            'URL': PHI.URL,
            'BIOID': PHI.Other,
            'HEALTHPLAN': PHI.Other
        }


def is_i2b2_2014(tag, rigid):
    if rigid:
        if tag in i2b2_2014.keys():
            return True
    else:
        if i2b2_2014[tag] in [PHI.Name, PHI.Address, PHI.Date]:
            return True
    return False


def change_to_enum(tag):
    return i2b2_2014.get(tag)


class Label:
    def __init__(self, tag, start, end):
        self.start = int(start)
        self.end = int(end)
        self.tag = change_to_enum(tag)
