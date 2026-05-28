from __future__ import annotations

import ipaddress
import re


DATE_LABELS = {
    "DATE_OF_BIRTH",
    "PASSPORT_EXPIRY",
    "PASSPORT_START_DATE",
    "MEDICARE_EXPIRY",
}

B_CLASS_LABELS = {
    "PERSON",
    "FIRST_NAME",
    "LAST_NAME",
    "ADDRESS",
    "ABORIGINALITY",
    "GENDER",
    "PRONOUN",
    "RELIGION_BELIEF",
    "RACIAL_ETHNIC_ORIGIN",
    "SEXUAL_ORIENTATION",
    "NATIONALITY",
    "CITIZENSHIP_STATUS",
    "MARITAL_STATUS",
    "MILITARY_VETERAN_STATUS",
    "CARING_RESPONSIBILITIES",
    "DISABILITY_OR_SPECIFIC_CONDITION",
    "MEDICAL_INFORMATION",
    "COUNSELLING_RECORDS",
    "MEDICAL_CERTIFICATE",
    "SPECIAL_CONSIDERATION",
    "CRIMINAL_RECORDS",
    "EMPLOYMENT_INFORMATION",
    "CONTRACT_TYPE",
    "SOCIO_ECONOMIC_STATUS",
    "NEXT_OF_KIN",
}


def _digits(value: str) -> str:
    return re.sub(r"\D+", "", value)


class FormatFilter:
    @staticmethod
    def get_candidates(value: str) -> set[str]:
        text = str(value).strip()
        compact = text.replace(" ", "").replace("-", "")
        digits = _digits(text)
        candidates = set(B_CLASS_LABELS)

        if FormatFilter.is_email(text):
            candidates.update({"EMAIL_ADDRESS", "WORK_EMAIL"})
        if FormatFilter.is_ip_address(text):
            candidates.add("IP_ADDRESS")
        if digits and len(digits) == 9:
            candidates.update({"AU_TFN", "STUDENT_ID", "EMPLOYEE_NUMBER", "PERSONNEL_NUMBER"})
        if digits and len(digits) == 6:
            candidates.update({"BSB", "STUDENT_ID", "EMPLOYEE_NUMBER", "UAC_ID", "PERSONNEL_NUMBER"})
        if digits and 6 <= len(digits) <= 10:
            candidates.update({"BANK_ACCOUNT_NUMBER", "STUDENT_ID", "EMPLOYEE_NUMBER", "PERSONNEL_NUMBER"})
        if digits and 13 <= len(digits) <= 19:
            candidates.add("PAYMENT_CARD_NUMBER")
        if re.fullmatch(r"\d{3,4}", digits):
            candidates.add("CREDIT_CARD_CVV")
        if FormatFilter.is_phone(text):
            candidates.update({"AU_PHONE", "MOBILE", "WORK_PHONE", "HOME_PHONE"})
        if FormatFilter.is_date_like(text):
            candidates.update(DATE_LABELS)
        if FormatFilter.is_credit_card_expiry_like(text):
            candidates.add("CREDIT_CARD_EXPIRY")
        if re.fullmatch(r"[A-Z0-9]{5,8}", compact, flags=re.IGNORECASE):
            candidates.update({"NUMBER_PLATE", "VEHICLE_REGO", "USERNAME"})
        if re.fullmatch(r"[A-Fa-f0-9]{16,64}", compact):
            candidates.add("HASHED_PAYMENT_CARD_NUMBER")
        if re.fullmatch(r"[-+]?\d{1,3}\.\d+", text):
            candidates.update({"LATITUDE", "LONGITUDE", "GEOLOCATION_INFORMATION"})
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{2,31}", text):
            candidates.update({"USERNAME", "SOCIAL_MEDIA_ID", "SOCIAL_MEDIA_ACCOUNT"})

        candidates.add("NON_PII")
        return candidates

    @staticmethod
    def is_email(value: str) -> bool:
        return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value.strip()))

    @staticmethod
    def is_ip_address(value: str) -> bool:
        try:
            ipaddress.ip_address(value.strip())
        except ValueError:
            return False
        return True

    @staticmethod
    def is_phone(value: str) -> bool:
        digits = _digits(value)
        return 8 <= len(digits) <= 15 and bool(re.search(r"(\+?61|04|\(?0\d\)?)", value))

    @staticmethod
    def is_date_like(value: str) -> bool:
        text = value.strip()
        month_names = (
            "jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
            "jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
        )
        return bool(
            re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", text)
            or re.fullmatch(r"\d{4}[./-]\d{1,2}[./-]\d{1,2}", text)
            or re.fullmatch(rf"(?i)(?:{month_names})\s+\d{{1,2}},?\s+\d{{4}}", text)
        )

    @staticmethod
    def is_credit_card_expiry_like(value: str) -> bool:
        return bool(re.fullmatch(r"(0?[1-9]|1[0-2])[/-]\d{2,4}", value.strip()))

    @staticmethod
    def rule_verified(label: str, value: str) -> bool:
        if label == "IP_ADDRESS":
            return FormatFilter.is_ip_address(value)
        if label == "PAYMENT_CARD_NUMBER":
            return FormatFilter._luhn_valid(_digits(value))
        if label == "AU_TFN":
            return FormatFilter._tfn_valid(_digits(value))
        if label == "BSB":
            return len(_digits(value)) == 6
        if label == "MEDICARE_NUMBER":
            return FormatFilter._medicare_valid(_digits(value))
        return False

    @staticmethod
    def _luhn_valid(digits: str) -> bool:
        if not digits.isdigit() or not (13 <= len(digits) <= 19):
            return False
        total = 0
        parity = len(digits) % 2
        for index, char in enumerate(digits):
            n = int(char)
            if index % 2 == parity:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        return total % 10 == 0

    @staticmethod
    def _tfn_valid(digits: str) -> bool:
        if not digits.isdigit() or len(digits) != 9:
            return False
        weights = [1, 4, 3, 7, 5, 8, 6, 9, 10]
        return sum(int(d) * w for d, w in zip(digits, weights)) % 11 == 0

    @staticmethod
    def _medicare_valid(digits: str) -> bool:
        if not digits.isdigit() or len(digits) < 10:
            return False
        weights = [1, 3, 7, 9, 1, 3, 7, 9]
        checksum = sum(int(digits[i]) * weights[i] for i in range(8)) % 10
        return checksum == int(digits[8])
