"""Pure-python tests for the model-agnostic core. No model loading."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from redaction.core import (
    Span, apply_policy, build_response, normalize_text,
    parse_annotated_output, redact_text, repair_offsets_to_input,
    resolve_overlaps, safe_postprocess_spans,
)
from redaction.core.postprocess import load_postprocess_rule_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_parse_annotated_output_simple() -> None:
    raw = 'Hi <pii type="PERSON">Alice</pii>, TFN <pii type="AU_TFN">123 456 789</pii>.'
    plain, spans = parse_annotated_output(raw)
    assert plain == "Hi Alice, TFN 123 456 789."
    assert [s.type for s in spans] == ["PERSON", "AU_TFN"]
    assert spans[0].value == "Alice"
    assert plain[spans[0].start:spans[0].end] == "Alice"
    assert plain[spans[1].start:spans[1].end] == "123 456 789"


def test_repair_offsets_to_input_unique_value() -> None:
    inp = "Please call Alice Wong on 0421 909 121."
    parsed = "Hi Alice Wong on 0421 909 121."
    spans = [
        Span(start=3, end=13, type="PERSON", value="Alice Wong"),
        Span(start=17, end=29, type="AU_PHONE", value="0421 909 121"),
    ]
    repaired, warns, did_repair = repair_offsets_to_input(inp, parsed, spans)
    assert did_repair is True
    assert any("round_trip_mismatch" in w for w in warns)
    assert inp[repaired[0].start:repaired[0].end] == "Alice Wong"
    assert inp[repaired[1].start:repaired[1].end] == "0421 909 121"


def test_resolve_overlaps_keeps_longer() -> None:
    spans = [
        Span(start=0, end=5, type="PERSON", value="Alice"),
        Span(start=0, end=12, type="PERSON", value="Alice Nguyen"),
    ]
    out = resolve_overlaps(spans)
    assert len(out) == 1
    assert out[0].end == 12


def test_apply_policy_uses_type_actions() -> None:
    spans = [
        Span(start=0, end=4, type="EMAIL", value="a@b.com"),
        Span(start=10, end=20, type="AU_TFN", value="123 456 789"),
    ]
    policy = {
        "default_action": "AUTO_REDACT",
        "type_actions": {"AU_TFN": "REVIEW"},
    }
    out = apply_policy(spans, policy)
    assert out[0].decision == "AUTO_REDACT"
    assert out[1].decision == "REVIEW"


def test_apply_policy_uses_confidence_thresholds() -> None:
    spans = [
        Span(start=0, end=5, type="PHONE", value="0421x", confidence=0.99),
        Span(start=10, end=20, type="PHONE", value="0421y", confidence=0.20),
    ]
    policy = {
        "default_action": "AUTO_REDACT",
        "type_actions": {"PHONE": "AUTO_REDACT"},
        "type_thresholds": {
            "PHONE": {"block_threshold": 0.5, "review_threshold": 0.3},
        },
    }
    out = apply_policy(spans, policy)
    assert out[0].decision == "AUTO_REDACT"
    assert out[1].decision == "PASS"


def test_apply_policy_handles_inverted_threshold_band() -> None:
    spans = [
        Span(start=0, end=5, type="PERSON", value="high", confidence=0.95),
        Span(start=10, end=15, type="PERSON", value="mid", confidence=0.50),
        Span(start=20, end=25, type="PERSON", value="low", confidence=0.10),
    ]
    policy = {
        "default_action": "AUTO_REDACT",
        "type_thresholds": {
            "PERSON": {"block_threshold": 0.30, "review_threshold": 0.90},
        },
    }
    out = apply_policy(spans, policy)
    assert [s.decision for s in out] == ["AUTO_REDACT", "REVIEW", "PASS"]


def test_redact_text_replace_with_tag() -> None:
    text = "Hi Alice."
    spans = [Span(start=3, end=8, type="PERSON", value="Alice", decision="AUTO_REDACT")]
    assert redact_text(text, spans) == "Hi [PERSON]."


def test_redact_text_mask() -> None:
    text = "Hi Alice."
    spans = [Span(start=3, end=8, type="PERSON", value="Alice", decision="AUTO_REDACT")]
    assert redact_text(text, spans, mode="mask") == "Hi *****."


def test_redact_text_leaves_review_spans_for_manual_review() -> None:
    text = "Card 4111 CVV 317."
    spans = [
        Span(start=5, end=9, type="PAYMENT_CARD_NUMBER", value="4111", decision="AUTO_REDACT"),
        Span(start=14, end=17, type="CREDIT_CARD_CVV", value="317", decision="REVIEW"),
    ]
    assert redact_text(text, spans) == "Card [PAYMENT_CARD_NUMBER] CVV 317."


def test_safe_postprocess_strips_dob_prefix() -> None:
    text = "DOB: 1990-01-02"
    spans = [Span(start=0, end=15, type="DATE_OF_BIRTH", value="DOB: 1990-01-02")]
    cleaned, _ = safe_postprocess_spans(text, spans, {"postprocess": {}})
    assert len(cleaned) == 1
    assert cleaned[0].value == "1990-01-02"
    assert text[cleaned[0].start:cleaned[0].end] == "1990-01-02"


def test_safe_postprocess_rescues_labeled_student_and_passport_identifiers() -> None:
    student = (
        "Student record for Amelia Chen. DOB 14/09/2002, USI L32K9P7H2Q, "
        "UAC ID 123456789, email amelia.chen@student.example.edu.au."
    )
    uac_start = student.index("123456789")
    student_cleaned, _ = safe_postprocess_spans(
        student,
        [Span(start=uac_start, end=uac_start + 9, type="IHI", value="123456789")],
        {"postprocess": {}},
    )
    assert [(s.type, s.value) for s in student_cleaned] == [
        ("DATE_OF_BIRTH", "14/09/2002"),
        ("USI", "L32K9P7H2Q"),
        ("UAC_ID", "123456789"),
        ("EMAIL", "amelia.chen@student.example.edu.au"),
    ]

    passport = (
        "Please verify Priya Nair at 44 Collins Street Melbourne VIC 3000. "
        "Passport PA1234567 expires 18/11/2030, driver licence D12345678, "
        "TFN 832 109 111."
    )
    passport_cleaned, _ = safe_postprocess_spans(passport, [], {"postprocess": {}})
    assert ("PASSPORT_EXPIRY", "18/11/2030") in [(s.type, s.value) for s in passport_cleaned]
    assert ("AU_TFN", "832 109 111") in [(s.type, s.value) for s in passport_cleaned]


def test_safe_postprocess_normalizes_common_taxonomy_aliases() -> None:
    text = (
        "BSB: 923-647\n"
        "Salary: $118,000\n"
        "insta @arjuntaylor_40\n"
        "next of kin is Fatima Brooks, phone (03) 9104 2944\n"
        "Passport expiry date: 07/2034\n"
        "Staff ID: E942830\n"
        "Medical detail: wrist fracture\n"
    )
    spans = [
        Span(start=text.index("923-647"), end=text.index("923-647") + 7, type="BSB", value="923-647"),
        Span(start=text.index("$118,000"), end=text.index("$118,000") + 8, type="SALARY_WAGE_EXPECTATION", value="$118,000"),
        Span(start=text.index("@arjuntaylor_40"), end=text.index("@arjuntaylor_40") + 15, type="SOCIAL_MEDIA_ACCOUNT", value="@arjuntaylor_40"),
        Span(start=text.index("Fatima Brooks"), end=text.index("Fatima Brooks") + 13, type="PERSON", value="Fatima Brooks"),
        Span(start=text.index("07/2034"), end=text.index("07/2034") + 7, type="PASSPORT_START_DATE", value="07/2034"),
        Span(start=text.index("E942830"), end=text.index("E942830") + 7, type="STUDENT_ID", value="E942830"),
        Span(start=text.index("wrist fracture"), end=text.index("wrist fracture") + 14, type="DISABILITY_OR_SPECIFIC_CONDITION", value="wrist fracture"),
    ]
    cleaned, _ = safe_postprocess_spans(text, spans, {"postprocess": {}})
    assert [(s.type, s.value) for s in cleaned] == [
        ("AU_BANK_ACCOUNT", "923-647"),
        ("SALARY", "$118,000"),
        ("SOCIAL_MEDIA_ID", "@arjuntaylor_40"),
        ("NEXT_OF_KIN", "Fatima Brooks"),
        ("PHONE", "(03) 9104 2944"),
        ("PASSPORT_EXPIRY", "07/2034"),
        ("EMPLOYEE_NUMBER", "E942830"),
        ("MEDICAL_INFORMATION", "wrist fracture"),
    ]


def test_safe_postprocess_rescues_contextual_structured_fields() -> None:
    text = (
        "SID: 426318590\n"
        "Staff ID: E942830\n"
        "Personnel Number: P74749731\n"
        "Vehicle REGO: AGL70L\n"
        "Number Plate: AGL70L\n"
        "IHI: 8003 0114 4880 7021\n"
        "Device ID: WIN-D97F-E38A-B4E9\n"
        "Website history flag: library-auth.example.test/profile\n"
        "Personal email: ella.wilson@student.example.test\n"
        "Medicare: 8259 87832 8\n"
        "Medical certificate says unfit from 02/05/2025 to 24/02/2026\n"
    )
    cleaned, _ = safe_postprocess_spans(text, [], {"postprocess": {}})
    pairs = [(s.type, s.value) for s in cleaned]
    assert ("STUDENT_ID", "426318590") in pairs
    assert ("EMPLOYEE_NUMBER", "E942830") in pairs
    assert ("PERSONNEL_NUMBER", "P74749731") in pairs
    assert pairs.count(("VEHICLE_ID", "AGL70L")) == 2
    assert ("IHI", "8003 0114 4880 7021") in pairs
    assert ("DEVICE_ID", "WIN-D97F-E38A-B4E9") in pairs
    assert ("WEBSITE_HISTORY", "library-auth.example.test/profile") in pairs
    assert ("EMAIL", "ella.wilson@student.example.test") in pairs
    assert ("MEDICARE_NUMBER", "8259 87832 8") in pairs
    assert ("MEDICAL_CERTIFICATE", "02/05/2025") in pairs
    assert ("MEDICAL_CERTIFICATE", "24/02/2026") in pairs


def test_safe_postprocess_rescues_identifier_verification_fields() -> None:
    text = (
        "Medicare expiry: 03/2032\n"
        "Medicare number is Medicare: 2938-4756-12-1. Card expiry: 08/2029\n"
        "National ID: NID-RS-420252-M\n"
        "Centrelink Reference Number: 948 686 159Q\n"
        "Passport Number: P8581172\n"
        "Passport start date: 14/08/2019\n"
        "Passport expiry date: 07/2034\n"
        "Scholarship ref: SCH-2026-8556Z\n"
    )
    cleaned, _ = safe_postprocess_spans(text, [], {"postprocess": {}})
    pairs = [(s.type, s.value) for s in cleaned]
    assert ("MEDICARE_EXPIRY", "03/2032") in pairs
    assert ("MEDICARE_EXPIRY", "08/2029") in pairs
    assert ("NATIONAL_IDENTITY_CARD", "NID-RS-420252-M") in pairs
    assert ("CENTRELINK_REFERENCE_NUMBER", "948 686 159Q") in pairs
    assert ("AU_PASSPORT", "P8581172") in pairs
    assert ("PASSPORT_START_DATE", "14/08/2019") in pairs
    assert ("PASSPORT_EXPIRY", "07/2034") in pairs
    assert ("SCHOLARSHIP", "SCH-2026-8556Z") in pairs


def test_safe_postprocess_rescues_messy_contextual_formats() -> None:
    text = (
        "bday 7.3.2004 I think\n"
        "sid: 5102 88411\n"
        "UAC no. 221 904 778\n"
        "mobile: 04 19 882 006\n"
        "acct: 0088 1992 44\n"
        "card used last time: 4111 9090 3333 1200 exp 08/29\n"
        "Vehicle Registration (License Plate): VIC-987-XYZ.\n"
    )
    spans = [
        Span(
            start=text.index("221 904 778"),
            end=text.index("221 904 778") + len("221 904 778"),
            type="PHONE",
            value="221 904 778",
            confidence=0.90,
        ),
    ]
    cleaned, _ = safe_postprocess_spans(text, spans, {"postprocess": {}})
    pairs = [(s.type, s.value) for s in cleaned]
    assert ("DATE_OF_BIRTH", "7.3.2004") in pairs
    assert ("STUDENT_ID", "5102 88411") in pairs
    assert ("UAC_ID", "221 904 778") in pairs
    assert ("PHONE", "04 19 882 006") in pairs
    assert ("AU_BANK_ACCOUNT", "0088 1992 44") in pairs
    assert ("PAYMENT_CARD_NUMBER", "4111 9090 3333 1200") in pairs
    assert ("CREDIT_CARD_EXPIRY", "08/29") in pairs
    assert ("VEHICLE_ID", "VIC-987-XYZ") in pairs
    assert ("PHONE", "221 904 778") not in pairs


def test_postprocess_registry_files_load_without_new_taxonomy() -> None:
    rules = load_postprocess_rule_registry(PROJECT_ROOT)
    labels = {rule.label for rule in rules}
    assert {
        "DATE_OF_BIRTH",
        "AU_BANK_ACCOUNT",
        "STUDENT_ID",
        "UAC_ID",
        "VEHICLE_ID",
        "CREDIT_CARD_EXPIRY",
        "MEDICARE_EXPIRY",
        "PERSON",
        "WEBSITE_HISTORY",
        "MEDICAL_CERTIFICATE",
        "RELIGION_BELIEF",
        "SOCIO_ECONOMIC_STATUS",
        "HASHED_PAYMENT_CARD_NUMBER",
        "CAMERA_FOOTAGE_AUDIO",
        "AUDIO_INFORMATION",
        "FACIAL_RECOGNITION",
        "FINGERPRINT",
        "VOICE_RECOGNITION",
        "SIGNATURE",
    }.issubset(labels)


def test_registry_driven_contextual_fallback_rules() -> None:
    text = (
        "bday 7.3.2004 I think\n"
        "sid: 5102 88411\n"
        "UAC no. 221 904 778\n"
        "acct: 0088 1992 44\n"
        "card used last time: 4111 9090 3333 1200 exp 08/29\n"
        "Medicare number is Medicare: 2938-4756-12-1. Card expiry: 08/2029\n"
        "Vehicle Registration (License Plate): VIC-987-XYZ.\n"
        "Also gave car rego: NSW CXT-72Q.\n"
    )
    cleaned, _ = safe_postprocess_spans(
        text,
        [],
        {"postprocess": {"add_contextual_identifier_spans": False, "add_registry_contextual_spans": True}},
    )
    pairs = [(s.type, s.value) for s in cleaned]
    assert ("DATE_OF_BIRTH", "7.3.2004") in pairs
    assert ("STUDENT_ID", "5102 88411") in pairs
    assert ("UAC_ID", "221 904 778") in pairs
    assert ("AU_BANK_ACCOUNT", "0088 1992 44") in pairs
    assert ("PAYMENT_CARD_NUMBER", "4111 9090 3333 1200") in pairs
    assert ("CREDIT_CARD_EXPIRY", "08/29") in pairs
    assert ("MEDICARE_EXPIRY", "08/2029") in pairs
    assert ("VEHICLE_ID", "VIC-987-XYZ") in pairs
    assert ("VEHICLE_ID", "NSW CXT-72Q") in pairs


def test_registry_driven_contextual_text_fields() -> None:
    text = (
        "Account name: Mia-Louise Martinez-Rivera\n"
        "Bank: Southern Mutual Bank\n"
        "Bank: Campus Mutual\n"
        "Role: lab demonstrator\n"
        "Contract type: continuing, part-time 0.8 FTE\n"
        "Citizenship Status: Permanent Resident\n"
        "Caring responsibilities: cares for younger sibling after school\n"
        "Postal address: Level 6, 76 High Street, Redfern NSW 2016\n"
        "Phoebe: new address is 7/68, 178 City Road, Northbridge WA 6003\n"
        "Fatima: next of kin is Tahlia Park, phone (08) 7786 4519\n"
        "Priya: pronouns he/they\n"
        "Reason: migraine + anxiety flare-up\n"
    )
    cleaned, _ = safe_postprocess_spans(
        text,
        [],
        {"postprocess": {"add_contextual_identifier_spans": False, "add_registry_contextual_spans": True}},
    )
    pairs = [(s.type, s.value) for s in cleaned]
    assert ("AU_BANK_ACCOUNT", "Mia-Louise Martinez-Rivera") in pairs
    assert ("AU_BANK_ACCOUNT", "Southern Mutual Bank") in pairs
    assert ("AU_BANK_ACCOUNT", "Campus Mutual") in pairs
    assert ("EMPLOYMENT_INFORMATION", "lab demonstrator") in pairs
    assert ("CONTRACT_TYPE", "continuing, part-time 0.8 FTE") in pairs
    assert ("CITIZENSHIP_STATUS", "Permanent Resident") in pairs
    assert ("CARING_RESPONSIBILITIES", "cares for younger sibling after school") in pairs
    assert ("ADDRESS", "Level 6, 76 High Street, Redfern NSW 2016") in pairs
    assert ("ADDRESS", "7/68, 178 City Road, Northbridge WA 6003") in pairs
    assert ("NEXT_OF_KIN", "Tahlia Park") in pairs
    assert ("PRONOUN", "he/they") in pairs
    assert ("MEDICAL_INFORMATION", "migraine + anxiety flare-up") in pairs


def test_registry_driven_finance_biometric_and_evidence_fields() -> None:
    text = (
        "acct 70929767\n"
        "account number 639 178 972\n"
        "refund account 0789378676\n"
        "Stored hashed card ref card_hash:4723b26560741ff3c4951cd9ae00ef2a.\n"
        "Payment hash sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n"
        "camera footage audio clip CCTV-2026-PF-4959\n"
        "audible conversation ref AUD-2026-42054\n"
        "facial recognition template FACE-XDB-990130\n"
        "fingerprint scan FP-4161-XOR-11\n"
        "voice recognition sample VOICE-JE-266631\n"
        "signature image ref SIG-2025-87634\n"
        "Pension Card: SEN-448812\n"
    )
    cleaned, _ = safe_postprocess_spans(
        text,
        [],
        {"postprocess": {"add_contextual_identifier_spans": False, "add_registry_contextual_spans": True}},
    )
    pairs = [(s.type, s.value) for s in cleaned]
    assert ("AU_BANK_ACCOUNT", "70929767") in pairs
    assert ("AU_BANK_ACCOUNT", "639 178 972") in pairs
    assert ("AU_BANK_ACCOUNT", "0789378676") in pairs
    assert ("HASHED_PAYMENT_CARD_NUMBER", "card_hash:4723b26560741ff3c4951cd9ae00ef2a") in pairs
    assert (
        "HASHED_PAYMENT_CARD_NUMBER",
        "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ) in pairs
    assert ("CAMERA_FOOTAGE_AUDIO", "CCTV-2026-PF-4959") in pairs
    assert ("AUDIO_INFORMATION", "AUD-2026-42054") in pairs
    assert ("FACIAL_RECOGNITION", "FACE-XDB-990130") in pairs
    assert ("FINGERPRINT", "FP-4161-XOR-11") in pairs
    assert ("VOICE_RECOGNITION", "VOICE-JE-266631") in pairs
    assert ("SIGNATURE", "SIG-2025-87634") in pairs
    assert ("PENSION_CARD_NUMBER", "SEN-448812") in pairs


def test_registry_evidence_fields_still_require_context() -> None:
    text = (
        "Invoice ref AUD-2026-42054\n"
        "Reference token CCTV-2026-PF-4959\n"
        "Ticket FACE-XDB-990130\n"
        "System-generated FP-4161-XOR-11\n"
        "Placeholder VOICE-JE-266631\n"
        "Public example SIG-2025-87634\n"
        "test token sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n"
        "bare SEN-448812\n"
    )
    cleaned, _ = safe_postprocess_spans(
        text,
        [],
        {"postprocess": {"add_contextual_identifier_spans": False, "add_registry_contextual_spans": True}},
    )
    assert cleaned == []


def test_context_normalization_relabels_account_and_personnel_conflicts() -> None:
    text = "acct 70929767\nPersonnel Number: P74749731\n"
    cleaned, _ = safe_postprocess_spans(
        text,
        [
            Span(start=text.index("70929767"), end=text.index("70929767") + 8, type="AU_DRIVERS_LICENCE", value="70929767", confidence=0.98),
            Span(start=text.index("P74749731"), end=text.index("P74749731") + 9, type="EMPLOYEE_NUMBER", value="P74749731", confidence=0.98),
        ],
        {"postprocess": {"add_contextual_identifier_spans": False, "add_registry_contextual_spans": True}},
    )
    assert [(s.type, s.value) for s in cleaned] == [
        ("AU_BANK_ACCOUNT", "70929767"),
        ("PERSONNEL_NUMBER", "P74749731"),
    ]


def test_registry_text_fields_still_require_context() -> None:
    text = (
        "Permanent Resident\n"
        "continuing, part-time 0.8 FTE\n"
        "lab demonstrator\n"
        "Southern Mutual Bank\n"
        "Level 6, 76 High Street, Redfern NSW 2016\n"
        "Tahlia Park\n"
        "he/they\n"
        "migraine + anxiety flare-up\n"
        "cares for younger sibling after school\n"
    )
    cleaned, _ = safe_postprocess_spans(
        text,
        [],
        {"postprocess": {"add_contextual_identifier_spans": False, "add_registry_contextual_spans": True}},
    )
    assert cleaned == []


def test_vehicle_context_conflicts_are_routed_to_review() -> None:
    text = "Vehicle Registration (License Plate): VIC987XYZ."
    start = text.index("VIC987XYZ")
    cleaned, _ = safe_postprocess_spans(
        text,
        [Span(start=start, end=start + len("VIC987XYZ"), type="AU_DRIVERS_LICENCE", value="VIC987XYZ", confidence=0.98)],
        {"postprocess": {}},
    )
    assert [(s.type, s.value, s.confidence) for s in cleaned] == [("VEHICLE_ID", "VIC987XYZ", 0.8)]
    assert "vehicle_context_label_conflict" in cleaned[0].postprocess

    policy = json.loads((PROJECT_ROOT / "configs" / "policies" / "opf-v3-default-v1.json").read_text())
    decided = apply_policy(cleaned, policy)
    assert [(s.type, s.decision) for s in decided] == [("VEHICLE_ID", "REVIEW")]

    rescued, _ = safe_postprocess_spans(text, [], {"postprocess": {}})
    decided = apply_policy(rescued, policy)
    assert [(s.type, s.value, s.decision) for s in decided] == [("VEHICLE_ID", "VIC987XYZ", "REVIEW")]


def test_registry_rules_suppress_hard_negative_and_bare_values() -> None:
    text = (
        "Invoice: 123456789\n"
        "Ticket: INC-0412-345-678\n"
        "Reference: UAC-123456789-TEST\n"
        "Reference token VIC-987-XYZ is not a plate.\n"
        "Meeting date 7.3.2004 is not DOB.\n"
        "Plain number 0088 1992 44 is not enough context.\n"
        "Code EXP 08/29 is not a card expiry without card context.\n"
        "Bare number 0412 345 678 is not enough context.\n"
    )
    cleaned, _ = safe_postprocess_spans(
        text,
        [],
        {"postprocess": {"add_contextual_identifier_spans": False, "add_registry_contextual_spans": True}},
    )
    assert cleaned == []


def test_safe_postprocess_does_not_globally_match_messy_formats() -> None:
    text = (
        "Reference token VIC-987-XYZ is not a plate.\n"
        "Meeting date 7.3.2004 is not DOB.\n"
        "Plain number 0088 1992 44 is not enough context.\n"
        "Code EXP 08/29 is not a card expiry without card context.\n"
    )
    cleaned, _ = safe_postprocess_spans(text, [], {"postprocess": {}})
    assert cleaned == []


def test_safe_postprocess_rescues_sensitive_attributes_and_group_people() -> None:
    text = (
        "1. Jacob Miller\n"
        "   SID: 458473748\n"
        "2. Lucas Brooks\n"
        "   Email: lucas.brooks81@student.example.test\n"
        "Student: Ella Mae Wilson\n"
        "Religion / Religious Beliefs: Buddhist\n"
        "Socio Economic Status: financial hardship category\n"
        "Reason: illness in group; one uploaded medical certificate.medical certificate\n"
    )
    cleaned, _ = safe_postprocess_spans(text, [], {"postprocess": {}})
    pairs = [(s.type, s.value) for s in cleaned]
    assert ("PERSON", "Jacob Miller") in pairs
    assert ("PERSON", "Lucas Brooks") in pairs
    assert ("PERSON", "Ella Mae Wilson") in pairs
    assert ("STUDENT_ID", "458473748") in pairs
    assert ("EMAIL", "lucas.brooks81@student.example.test") in pairs
    assert ("RELIGION_BELIEF", "Buddhist") in pairs
    assert ("SOCIO_ECONOMIC_STATUS", "financial hardship category") in pairs
    assert ("MEDICAL_CERTIFICATE", "medical certificate") in pairs
    assert [s.start for s in cleaned if s.type == "MEDICAL_CERTIFICATE"] == [text.rindex("medical certificate")]


def test_safe_postprocess_promotes_existing_rule_matched_model_span() -> None:
    text = "SID: 458473748\nEmail: lucas.brooks81@student.example.test\nMobile: 0456 323 671"
    spans = [
        Span(
            start=text.index("458473748"),
            end=text.index("458473748") + 9,
            type="STUDENT_ID",
            value="458473748",
            confidence=0.70,
        ),
        Span(
            start=text.index("lucas.brooks81"),
            end=text.index("lucas.brooks81") + len("lucas.brooks81@student.example.test"),
            type="EMAIL",
            value="lucas.brooks81@student.example.test",
            confidence=0.60,
        ),
        Span(
            start=text.index("0456"),
            end=text.index("0456") + len("0456 323 671"),
            type="PHONE",
            value="0456 323 671",
            confidence=0.80,
        ),
    ]
    cleaned, _ = safe_postprocess_spans(text, spans, {"postprocess": {}})
    assert [(s.type, s.value, s.source, s.confidence) for s in cleaned] == [
        ("STUDENT_ID", "458473748", "rule", 1.0),
        ("EMAIL", "lucas.brooks81@student.example.test", "rule", 1.0),
        ("PHONE", "0456 323 671", "rule", 1.0),
    ]


def test_safe_postprocess_rule_spans_override_overlapping_model_candidates() -> None:
    text = "Medical certificate says unfit from 02/05/2025 to 24/02/2026"
    spans = [
        Span(
            start=text.index("02/05/2025"),
            end=text.index("24/02/2026") + len("24/02/2026"),
            type="DATE_OF_BIRTH",
            value="02/05/2025 to 24/02/2026",
            confidence=0.1,
            source="model",
        )
    ]
    cleaned, _ = safe_postprocess_spans(text, spans, {"postprocess": {}})
    assert [(s.type, s.value, s.source) for s in cleaned] == [
        ("MEDICAL_CERTIFICATE", "02/05/2025", "rule"),
        ("MEDICAL_CERTIFICATE", "24/02/2026", "rule"),
    ]


def test_safe_postprocess_drops_common_hard_negative_false_positives() -> None:
    text = (
        "Invoice: INV-5101-7351\n"
        "Receipt no REC-733358-669\n"
        "Claim number: WC-2025-03980\n"
        "Placeholder email: user@example.test\n"
        "Test token: tok_4111111111111111\n"
    )
    spans = [
        Span(start=text.index("INV-5101-7351"), end=text.index("INV-5101-7351") + 13, type="AU_DRIVERS_LICENCE", value="INV-5101-7351"),
        Span(start=text.index("REC-733358-669"), end=text.index("REC-733358-669") + 14, type="AU_DRIVERS_LICENCE", value="REC-733358-669"),
        Span(start=text.index("WC-2025-03980"), end=text.index("WC-2025-03980") + 13, type="AU_DRIVERS_LICENCE", value="WC-2025-03980"),
        Span(start=text.index("user@example.test"), end=text.index("user@example.test") + 17, type="EMAIL", value="user@example.test"),
        Span(start=text.index("4111111111111111"), end=text.index("4111111111111111") + 16, type="PAYMENT_CARD_NUMBER", value="4111111111111111"),
    ]
    cleaned, _ = safe_postprocess_spans(text, spans, {"postprocess": {}})
    assert [(s.type, s.value) for s in cleaned] == [("WORKERS_COMPENSATION_CLAIM", "WC-2025-03980")]

    real = "Personal email: ella.wilson@student.example.test"
    start = real.index("ella")
    cleaned, _ = safe_postprocess_spans(
        real,
        [Span(start=start, end=len(real), type="EMAIL", value=real[start:])],
        {"postprocess": {}},
    )
    assert [(s.type, s.value) for s in cleaned] == [("EMAIL", "ella.wilson@student.example.test")]


def test_safe_postprocess_drops_uac_and_numeric_hard_negative_false_positives() -> None:
    text = (
        "Invoice: 123456789\n"
        "Reference: UAC-123456789-TEST\n"
        "permit ref PARK-54817742 is system generated\n"
        "Actual student below:\n"
        "UAC ID: 251066238\n"
    )
    spans = [
        Span(start=text.index("123456789"), end=text.index("123456789") + 9, type="AU_BANK_ACCOUNT", value="123456789"),
        Span(start=text.index("123456789-TEST"), end=text.index("123456789-TEST") + 14, type="UAC_ID", value="123456789-TEST"),
        Span(start=text.index("PARK-54817742"), end=text.index("PARK-54817742") + 13, type="AU_DRIVERS_LICENCE", value="PARK-54817742"),
    ]
    cleaned, _ = safe_postprocess_spans(text, spans, {"postprocess": {}})
    assert [(s.type, s.value) for s in cleaned] == [("UAC_ID", "251066238")]


def test_build_response_metadata() -> None:
    text = "x"
    payload = build_response(
        text=text, spans=[], policy={
            "policy_id": "p", "model_version": "m",
            "taxonomy_version": "t", "schema_version": "redaction-output-v1",
        },
        raw_offset_mapping_applied=False, warnings=[],
    )
    assert payload["metadata"]["policy_id"] == "p"
    assert payload["metadata"]["normalization"] == "NFC"
    assert payload["redacted_text"] == "x"
    assert payload["spans"] == []


def test_build_response_omits_raw_span_values() -> None:
    text = "Hi Alice."
    spans = [Span(start=3, end=8, type="PERSON", value="Alice", decision="AUTO_REDACT")]
    payload = build_response(
        text=text,
        spans=spans,
        policy={
            "policy_id": "p",
            "model_version": "m",
            "taxonomy_version": "t",
            "schema_version": "redaction-output-v1",
        },
        raw_offset_mapping_applied=False,
        warnings=[],
    )
    assert payload["redacted_text"] == "Hi [PERSON]."
    assert payload["spans"][0]["type"] == "PERSON"
    assert "value" not in payload["spans"][0]


def test_build_response_hides_pass_spans() -> None:
    text = "Name Alice token maybe."
    spans = [
        Span(start=5, end=10, type="PERSON", value="Alice", decision="AUTO_REDACT"),
        Span(start=11, end=16, type="USERNAME", value="token", decision="REVIEW"),
        Span(start=17, end=22, type="DEVICE_ID", value="maybe", decision="PASS"),
    ]
    payload = build_response(
        text=text,
        spans=spans,
        policy={
            "policy_id": "p",
            "model_version": "m",
            "taxonomy_version": "t",
            "schema_version": "redaction-output-v1",
        },
        raw_offset_mapping_applied=False,
        warnings=[],
    )
    assert payload["redacted_text"] == "Name [PERSON] token maybe."
    assert [s["decision"] for s in payload["spans"]] == ["AUTO_REDACT", "REVIEW"]


def test_build_response_masks_configured_review_types_in_redacted_text() -> None:
    text = "USI Q7XH22PL9A phone 0412 345 678 note low."
    spans = [
        Span(start=4, end=14, type="USI", value="Q7XH22PL9A", decision="REVIEW"),
        Span(start=21, end=33, type="PHONE", value="0412 345 678", decision="REVIEW"),
        Span(start=39, end=42, type="PRONOUN", value="low", decision="REVIEW"),
    ]
    payload = build_response(
        text=text,
        spans=spans,
        policy={
            "policy_id": "p",
            "model_version": "m",
            "taxonomy_version": "t",
            "schema_version": "redaction-output-v1",
            "redact_review_types": ["USI", "PHONE"],
        },
        raw_offset_mapping_applied=False,
        warnings=[],
    )
    assert payload["redacted_text"] == "USI [USI] phone [PHONE] note low."
    assert [s["decision"] for s in payload["spans"]] == ["REVIEW", "REVIEW", "REVIEW"]


def test_backend_configs_use_schema_types() -> None:
    schema = json.loads((PROJECT_ROOT / "schemas" / "redaction-output-v1.schema.json").read_text())
    allowed = set(schema["$defs"]["pii_type"]["enum"])
    for path in sorted((PROJECT_ROOT / "configs" / "backends").glob("*.json")):
        cfg = json.loads(path.read_text())
        unsupported = sorted(set(cfg["supported_types"]) - allowed)
        assert unsupported == [], f"{path.name} unsupported by schema: {unsupported[:8]}"


def test_schema_exposes_only_actionable_decisions() -> None:
    schema = json.loads((PROJECT_ROOT / "schemas" / "redaction-output-v1.schema.json").read_text())
    assert schema["$defs"]["decision"]["enum"] == ["AUTO_REDACT", "REVIEW"]
    assert "HASHED_PAYMENT_CARD_NUMBER" in schema["$defs"]["pii_type"]["enum"]


def test_opf_clear_payment_fields_reach_auto_redact() -> None:
    policy = json.loads((PROJECT_ROOT / "configs" / "policies" / "opf-v3-default-v1.json").read_text())
    spans = [
        Span(start=0, end=19, type="PAYMENT_CARD_NUMBER", value="4111 1111 1111 1111", confidence=0.985),
        Span(start=20, end=25, type="CREDIT_CARD_EXPIRY", value="08/29", confidence=0.976),
        Span(start=26, end=29, type="CREDIT_CARD_CVV", value="317", confidence=0.961),
    ]
    out = apply_policy(spans, policy)
    assert [s.decision for s in out] == ["AUTO_REDACT", "AUTO_REDACT", "AUTO_REDACT"]


def test_opf_clear_australian_ids_reach_auto_redact() -> None:
    policy = json.loads((PROJECT_ROOT / "configs" / "policies" / "opf-v3-default-v1.json").read_text())
    spans = [
        Span(start=0, end=9, type="AU_PASSPORT", value="PA1234567", confidence=0.972),
        Span(start=10, end=21, type="PASSPORT_EXPIRY", value="18/11/2030", confidence=0.95),
        Span(start=22, end=33, type="AU_TFN", value="832 109 111", confidence=0.95),
        Span(start=34, end=43, type="UAC_ID", value="123456789", confidence=0.95),
    ]
    out = apply_policy(spans, policy)
    assert [s.decision for s in out] == ["AUTO_REDACT", "AUTO_REDACT", "AUTO_REDACT", "AUTO_REDACT"]


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
