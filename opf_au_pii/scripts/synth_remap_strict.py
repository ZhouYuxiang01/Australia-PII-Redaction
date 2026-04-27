"""Remap synthetic-data type aliases to the official 73-class names, drop any
remaining unknown types. Run after synth_filter.py.
"""
import argparse, json, os
from collections import Counter

DEFAULT_IN  = "/home/admin/ZYX/opf_au_pii/data/processed/data_opf_v3/synth_opf_clean.jsonl"
DEFAULT_OUT = "/home/admin/ZYX/opf_au_pii/data/processed/data_opf_v3/synth_opf_strict.jsonl"
DEFAULT_LS  = "/home/admin/ZYX/opf_au_pii/configs/custom_label_space_73.v1.1.1.json"

REMAP = {
    # Passport variants
    "PASSPORT_NUMBER": "AU_PASSPORT",
    "PASSPORT": "AU_PASSPORT",
    "AU_PASSPORT_NUMBER": "AU_PASSPORT",
    # Driver licence variants
    "DRIVERS_LICENSE": "AU_DRIVERS_LICENCE",
    "DRIVERS_LICENCE": "AU_DRIVERS_LICENCE",
    "DRIVER_LICENCE": "AU_DRIVERS_LICENCE",
    "DRIVER_LICENSE": "AU_DRIVERS_LICENCE",
    "AU_DRIVER_LICENCE": "AU_DRIVERS_LICENCE",
    "AU_DRIVER_LICENSE": "AU_DRIVERS_LICENCE",
    "DRIVERS_LICENCE_NUMBER": "AU_DRIVERS_LICENCE",
    "DRIVERS_LICENSE_NUMBER": "AU_DRIVERS_LICENCE",
    "DRIVER_LICENCE_NUMBER": "AU_DRIVERS_LICENCE",
    "AU_DRIVER_LICENCE_NUMBER": "AU_DRIVERS_LICENCE",
    # Bank / payments
    "CREDIT_CARD_NUMBER": "PAYMENT_CARD_NUMBER",
    "BANK_ACCOUNT_NUMBER": "AU_BANK_ACCOUNT",
    "ACCOUNT_NUMBER": "AU_BANK_ACCOUNT",
    "AU_BSB": "BSB",
    # Phone
    "LANDLINE": "PHONE",
    "AU_PHONE": "PHONE",
    # Employment
    "EMPLOYEE_ID": "EMPLOYEE_NUMBER",
    # Sexual orientation typo
    "PERSONAL_ORIENTATION": "SEXUAL_ORIENTATION",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", default=DEFAULT_IN)
    ap.add_argument("--out_path", default=DEFAULT_OUT)
    ap.add_argument("--label_space", default=DEFAULT_LS)
    args = ap.parse_args()

    ls = json.load(open(args.label_space))
    allowed = set()
    for k, v in ls.items():
        if isinstance(v, list):
            allowed.update(v)
    allowed.discard("O")
    print(f"allowed types: {len(allowed)}")

    n_in = 0
    n_kept_doc = 0
    n_dropped_doc = 0
    n_remapped = 0
    n_unknown_drop = 0
    n_kept = 0
    remap_counts = Counter()
    drop_counts = Counter()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.in_path) as f, open(args.out_path, "w") as out:
        for line in f:
            n_in += 1
            r = json.loads(line)
            text = r["text"]
            new_spans = {}
            for k, ranges in r.get("spans", {}).items():
                if ":" not in k:
                    continue
                typ, val = k.split(":", 1)
                typ = typ.strip()
                val = val.strip()
                if typ in REMAP:
                    typ = REMAP[typ]
                    n_remapped += len(ranges)
                    remap_counts[typ] += len(ranges)
                if typ not in allowed:
                    n_unknown_drop += len(ranges)
                    drop_counts[typ] += len(ranges)
                    continue
                new_key = f"{typ}: {val}"
                # Merge if a remap collides with existing key
                if new_key in new_spans:
                    new_spans[new_key].extend(ranges)
                else:
                    new_spans[new_key] = list(ranges)
                n_kept += len(ranges)
            if new_spans:
                out.write(json.dumps({"example_id": r["example_id"], "text": text,
                                      "spans": new_spans}, ensure_ascii=False) + "\n")
                n_kept_doc += 1
            else:
                n_dropped_doc += 1

    print(f"in: {n_in} docs  kept: {n_kept_doc}  dropped (no spans left): {n_dropped_doc}")
    print(f"spans: kept={n_kept}  remapped={n_remapped}  dropped_unknown={n_unknown_drop}")
    if remap_counts:
        print("remap targets:")
        for t, n in remap_counts.most_common():
            print(f"  -> {t}: {n}")
    if drop_counts:
        print("dropped unknown types:")
        for t, n in drop_counts.most_common():
            print(f"  {t}: {n}")
    print(f"\nwrote {args.out_path}")


if __name__ == "__main__":
    main()
