"""Run the long sparse-PII test through the 4B hybrid backend; dump every span."""
import sys, os, json
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
sys.path.insert(0, '/home/admin/ZYX/redaction-wrapper')

from redaction.backends.registry import build_backend_from_path
from redaction.core import (apply_policy, build_response, load_json,
                             normalize_text, safe_postprocess_spans)

TEXT = """Case note from the late afternoon walk-in session:

The student arrived about twenty minutes before closing and said the issue was not urgent, but she wanted someone to record the details before she forgot. She was mainly asking about a timetable clash, a group assignment dispute, and whether the replacement tutorial would count toward attendance. Most of the discussion was about course planning, not identity or finance. She mentioned that several dates in her notes were just room bookings, for example room 14/09/2002 in Building A and booking reference 221 904 778 for the study pod, and she specifically said those were not personal identifiers.

Her name on the enrolment record should be Mia-Louise Tran, although she sometimes writes "M Tran" on informal forms. Near the end of the appointment she wrote down her student number as SID 5102 88411 and said the preferred student email was mia.tran04@student.example.edu.au. She also gave a mobile number, 0419 882 006, but asked that it only be used if the email bounces. The address she gave was 12B / 8 Wattle Street, Newtown NSW 2042.

There were several unrelated numbers in the same note. The staff member wrote "ticket id INC-0412-345-678" for the service desk case, but that is not a phone number. Another line says invoice 123456789 for a printer refund; this is an internal invoice, not a bank account. A sample email user@example.com appears in the troubleshooting template and should be treated as a placeholder. The line "fake card test token: tok_4111111111111111" was copied from a software testing note and should not be treated as a real payment card.

The student then explained that the refund had been delayed. For the actual refund, she gave BSB 062-001 and account number 123456789. She said the last card used on the portal ended in a full test-looking string, but this one was written as 4111 9090 3333 1200 with expiry 08/29. She also mentioned that the casual salary estimate on an old form was $92,500 plus loading, but the staff member was unsure whether that figure was still current.

In the identity check section, the handwritten copy was messy. It listed passport M8842190, IHI 8003 6012 4516 7791, and Centrelink reference CRN 455 220 991A. There was also a line saying "UAC no. 221 904 778" and another line saying "USI = Q7XH-22PL-9A". The student's date of birth was written as 07/03/2004. A separate note says "room date 07/03/2004 on the booking sheet", but that second occurrence was about a room booking and not a birthday.

For emergency contact, she said to call Grace Tran only if there is a serious problem. The number written beside that name was 02 9188 4410. She also wrote down vehicle rego NSW CXT-72Q because the parking office had asked for it, but the parking ticket reference PARK-2024-000778 should not be redacted as a vehicle registration. The staff member added that no medical certificate was sighted, no sanctions were discussed, and no disciplinary finding was recorded."""


def fmt_span(span, text):
    val = text[span.start:span.end].replace('\n', '\\n')
    if len(val) > 35:
        val = val[:32] + '...'
    return (f"  [{span.start:5d}-{span.end:<5d}] {span.type:>30s} | dec={span.decision or '<none>':>7s} "
            f"src={(span.source or '?'):<6s} conf={(span.confidence or 0):.3f} | {val!r:38s} | "
            f"reason={span.decision_reason or ''} pp={span.postprocess}")


def main():
    backend = build_backend_from_path('/home/admin/ZYX/redaction-wrapper/configs/backends/hybrid-opf-qwen4b.json')
    policy = load_json('/home/admin/ZYX/redaction-wrapper/configs/policies/hybrid-80class-v2-4b.json')

    text = normalize_text(TEXT)
    print(f'text length: {len(text)}\n')

    spans, diag = backend.detect_spans(text)
    print(f'=== STAGE 1: backend.detect_spans → {len(spans)} spans ===')
    for s in sorted(spans, key=lambda x: x.start):
        print(fmt_span(s, text))

    spans2, warnings = safe_postprocess_spans(text, spans, policy)
    print(f'\n=== STAGE 2: after safe_postprocess_spans → {len(spans2)} spans ===')
    for s in sorted(spans2, key=lambda x: x.start):
        print(fmt_span(s, text))

    spans3 = apply_policy(spans2, policy)
    print(f'\n=== STAGE 3: after apply_policy → {len(spans3)} spans ===')
    for s in sorted(spans3, key=lambda x: x.start):
        print(fmt_span(s, text))

    from redaction.core.policy import redact_text
    final = redact_text(text, spans3, mode='replace_with_tag')
    print(f'\n=== FINAL REDACTED TEXT ===\n{final}\n')

    print('=== LEAK CHECK ===')
    for needle in ['M8842190','5102 88411','Q7XH-22PL-9A','455 220 991A','8003 6012 4516 7791',
                   '4111 9090 3333 1200','0419 882 006','mia.tran04','user@example.com',
                   'tok_4111111111111111','UAC no. 221 904 778','PARK-2024-000778']:
        leaked = needle in final
        marker = 'LEAK' if leaked else '  ok'
        print(f'  {marker}: {needle!r}')

    # focus on user's failure cases
    print('\n=== FOCUS: ChatGPT failure cases ===')
    targets = [
        ('SID 5102 88411', 'STUDENT_ID', 'REDACT'),
        ('passport M8842190', 'AU_PASSPORT', 'REDACT'),
        ('NSW: 2098 771 334', 'AU_DRIVERS_LICENCE', 'REDACT'),
        ('IHI 8003 6012 4516 7791', 'IHI', 'REDACT'),
        ('CRN 455 220 991A', 'CENTRELINK_REFERENCE_NUMBER', 'REDACT'),
        ('UAC no. 221 904 778', 'UAC_ID', 'REDACT'),
        ('USI = Q7XH-22PL-9A', 'USI', 'REDACT'),
        ('booking reference 221 904 778', 'NON_PII', 'IGNORE'),
        ('parking ticket reference PARK-2024-000778', 'NON_PII', 'IGNORE'),
        ('12B / 8 Wattle Street, Newtown NSW 2042', 'ADDRESS', 'REDACT'),
    ]
    for needle, expected_type, expected_decision in targets:
        idx = text.find(needle)
        if idx < 0:
            print(f'  NOT FOUND in text: {needle!r}')
            continue
        end = idx + len(needle)
        hit = [s for s in spans3 if not (s.end <= idx or s.start >= end)]
        status = 'OK' if any(s.type == expected_type and (s.decision or '').lower() in (expected_decision.lower(), 'auto_redact' if expected_decision == 'REDACT' else expected_decision.lower()) for s in hit) else 'FAIL'
        print(f'\n  [{status}] {needle!r}  expected={expected_type}/{expected_decision}')
        for s in hit:
            print('    ' + fmt_span(s, text).strip())


if __name__ == '__main__':
    main()
