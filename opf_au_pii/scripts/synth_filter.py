"""Filter synthesized OPF training records, dropping spans that obviously violate
their declared type's surface form. Drops bad spans only; the rest of the doc is
kept. Reports per-type drop rate and writes a cleaned JSONL.
"""
import argparse, json, os, re, sys
from collections import Counter

DEFAULT_IN  = "/home/admin/ZYX/opf_au_pii/data/processed/data_opf_v3/synth_opf.jsonl"
DEFAULT_OUT = "/home/admin/ZYX/opf_au_pii/data/processed/data_opf_v3/synth_opf_clean.jsonl"

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
DIGIT_GROUP_RE = re.compile(r"^[\d\s\-]{6,}$")
AU_TFN_RE   = re.compile(r"^\d[\d\s\-]{7,}\d$")  # 9 digits with optional separators
BSB_RE      = re.compile(r"^\d{3}[\s\-]?\d{3}$")
PHONE_RE    = re.compile(r"^[+\d][\d\s\-()]{6,}$")
URL_LIKE    = re.compile(r"https?://|www\.|\.com|\.au|/in/|/\w+\.\w")
LATLON_RE   = re.compile(r"^-?\d{1,3}\.\d+$")
IP_RE       = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?:/\d{1,2})?$|^[0-9a-fA-F:]+::[0-9a-fA-F:]*$")


def violations(t: str, v: str) -> str:
    """Return reason string if value violates type's form; '' if OK."""
    vs = v.strip()
    if not vs:
        return "empty"
    if t == "EMAIL":
        if not EMAIL_RE.match(vs):
            return "email_missing_at_or_domain"
    elif t == "AU_TFN":
        digits = re.sub(r"\D", "", vs)
        if len(digits) != 9:
            return f"tfn_digit_count={len(digits)}"
    elif t == "BSB":
        if not BSB_RE.match(vs):
            digits = re.sub(r"\D", "", vs)
            if len(digits) != 6:
                return "bsb_not_6_digits"
    elif t == "PHONE":
        digits = re.sub(r"\D", "", vs)
        if len(digits) < 8 or len(digits) > 14:
            return f"phone_digit_count={len(digits)}"
        if "@" in vs:
            return "phone_has_at"
    elif t == "IP_ADDRESS":
        if not IP_RE.match(vs):
            return "ip_format"
    elif t == "LATITUDE":
        if not LATLON_RE.match(vs):
            return "lat_not_decimal"
        try:
            x = float(vs)
            if x < -90 or x > 90:
                return "lat_out_of_range"
        except Exception:
            return "lat_unparseable"
    elif t == "LONGITUDE":
        if not LATLON_RE.match(vs):
            return "lon_not_decimal"
        try:
            x = float(vs)
            if x < -180 or x > 180:
                return "lon_out_of_range"
        except Exception:
            return "lon_unparseable"
    elif t in {"CREDIT_CARD_EXPIRY", "MEDICARE_EXPIRY", "PASSPORT_EXPIRY",
               "PASSPORT_START_DATE"}:
        # MM/YY or MM/YYYY or DD/MM/YYYY etc — at minimum has digits and a separator
        digits = re.sub(r"\D", "", vs)
        if len(digits) < 4 or len(digits) > 8:
            return f"expiry_digit_count={len(digits)}"
        if not re.search(r"[/\-.]", vs):
            return "expiry_no_separator"
    elif t == "DATE_OF_BIRTH":
        digits = re.sub(r"\D", "", vs)
        if len(digits) < 6 or len(digits) > 8:
            return f"dob_digit_count={len(digits)}"
    # URL-like things in person/identity types
    if t in {"PERSON", "NEXT_OF_KIN", "AU_TFN", "BSB", "AU_DRIVERS_LICENCE",
             "STUDENT_ID", "EMPLOYEE_NUMBER", "PERSONNEL_NUMBER", "CENTRELINK_REFERENCE_NUMBER",
             "AU_BANK_ACCOUNT", "MEDICARE_NUMBER", "AU_PASSPORT", "PHONE"}:
        if URL_LIKE.search(vs):
            return f"{t.lower()}_looks_like_url"
        if "@" in vs and t != "PHONE":
            return f"{t.lower()}_has_at"
    # SOCIAL_MEDIA_ACCOUNT vs HISTORY heuristic: a single URL or handle is ACCOUNT, not HISTORY
    if t == "SOCIAL_MEDIA_HISTORY":
        # single token, no list/comma → likely should be account
        if "," not in vs and ";" not in vs and "\n" not in vs and len(vs) < 60:
            return "social_media_history_singleton_likely_account"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default=DEFAULT_IN)
    ap.add_argument("--out_path", default=DEFAULT_OUT)
    ap.add_argument("--report", action="store_true",
                    help="report only, do not write output")
    args = ap.parse_args()

    drop_reasons = Counter()
    drop_by_type = Counter()
    kept_by_type = Counter()
    n_in = 0
    n_kept_doc = 0
    n_dropped_doc = 0

    out_f = None
    if not args.report:
        os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
        out_f = open(args.out_path, "w")

    with open(args.in_path) as f:
        for line in f:
            n_in += 1
            r = json.loads(line)
            text = r["text"]
            new_spans = {}
            for k, ranges in r.get("spans", {}).items():
                if ":" in k:
                    typ = k.split(":", 1)[0].strip()
                    val = k.split(":", 1)[1].strip()
                else:
                    typ, val = k, ""
                reason = violations(typ, val)
                if reason:
                    drop_reasons[f"{typ}::{reason}"] += 1
                    drop_by_type[typ] += len(ranges)
                    continue
                new_spans[k] = ranges
                kept_by_type[typ] += len(ranges)
            if new_spans:
                if out_f:
                    out_f.write(json.dumps({"example_id": r["example_id"],
                                            "text": text, "spans": new_spans},
                                           ensure_ascii=False) + "\n")
                n_kept_doc += 1
            else:
                n_dropped_doc += 1

    if out_f:
        out_f.close()

    n_total_spans = sum(kept_by_type.values()) + sum(drop_by_type.values())
    n_dropped = sum(drop_by_type.values())
    print(f"in: {n_in} docs  kept: {n_kept_doc}  dropped (no spans left): {n_dropped_doc}")
    print(f"total spans: {n_total_spans}  kept: {sum(kept_by_type.values())}  "
          f"dropped: {n_dropped}  drop_rate: {100*n_dropped/max(n_total_spans,1):.2f}%")
    print()
    print("top drop reasons:")
    for r, n in drop_reasons.most_common(15):
        print(f"  {r}  {n}")
    print()
    print("drops by type (top 10):")
    for t, n in drop_by_type.most_common(10):
        kept = kept_by_type.get(t, 0)
        rate = n / max(kept + n, 1) * 100
        print(f"  {t:32s}  dropped={n:4d}  kept={kept:4d}  drop_rate={rate:5.1f}%")
    if not args.report:
        print(f"\nwrote {args.out_path}")


if __name__ == "__main__":
    main()
