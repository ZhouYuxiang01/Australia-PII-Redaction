import unittest

from pii_prep.format_filter import FormatFilter


class FormatFilterTests(unittest.TestCase):
    def test_numeric_identifier_candidates_include_bsb_and_non_pii(self):
        candidates = FormatFilter.get_candidates("123456")

        self.assertIn("BSB", candidates)
        self.assertIn("STUDENT_ID", candidates)
        self.assertIn("EMPLOYEE_NUMBER", candidates)
        self.assertIn("NON_PII", candidates)

    def test_ip_rule_verification_rejects_out_of_range_octets(self):
        self.assertTrue(FormatFilter.rule_verified("IP_ADDRESS", "242.30.143.150"))
        self.assertFalse(FormatFilter.rule_verified("IP_ADDRESS", "999.30.143.150"))

    def test_luhn_rule_verification_for_payment_cards(self):
        self.assertTrue(FormatFilter.rule_verified("PAYMENT_CARD_NUMBER", "4111 1111 1111 1111"))
        self.assertFalse(FormatFilter.rule_verified("PAYMENT_CARD_NUMBER", "4111 1111 1111 1112"))

    def test_full_date_candidates_include_date_labels_but_not_card_expiry(self):
        candidates = FormatFilter.get_candidates("04/05/1998")

        self.assertIn("DATE_OF_BIRTH", candidates)
        self.assertIn("PASSPORT_EXPIRY", candidates)
        self.assertIn("PASSPORT_START_DATE", candidates)
        self.assertIn("MEDICARE_EXPIRY", candidates)
        self.assertIn("NON_PII", candidates)
        self.assertNotIn("CREDIT_CARD_EXPIRY", candidates)

    def test_month_year_candidates_include_card_expiry(self):
        candidates = FormatFilter.get_candidates("04/2028")

        self.assertIn("CREDIT_CARD_EXPIRY", candidates)
        self.assertIn("NON_PII", candidates)

    def test_iso_and_month_name_dates_are_date_like(self):
        self.assertIn("DATE_OF_BIRTH", FormatFilter.get_candidates("1998-05-04"))
        self.assertIn("DATE_OF_BIRTH", FormatFilter.get_candidates("May 04 1998"))


if __name__ == "__main__":
    unittest.main()
