"""Produce a v2b vs v3 delta report on external_1000."""
import json
from collections import Counter

V2B = "/home/admin/ZYX/opf_au_pii/runs/final/opf_73class_v2b_full/external_1000"
V3  = "/home/admin/ZYX/opf_au_pii/runs/final/opf_73class_v3_full/external_1000"


def load_char_metrics(d):
    return json.load(open(f"{d}/positive_char_metrics.json"))


def opf_metrics(path):
    """Pull OPF span-level metrics from positive_metrics.json (it nests under 'official')."""
    m = json.load(open(path))
    out = {}
    if "detection" in m:
        out["detection"] = {k: m["detection"].get(k) for k in ("precision","recall","f1")}
    if "span" in m:
        out["span"] = {k: m["span"].get(k) for k in ("precision","recall","f1")}
    return out


def per_label(d):
    m = load_char_metrics(d)
    typed = m["typed"]
    tp = typed["tp_by_label"]
    fp = typed.get("fp_by_label", {})
    fn = typed.get("fn_by_label", {})
    out = {}
    for L in sorted(set(tp) | set(fp) | set(fn)):
        t = tp.get(L,0); f = fp.get(L,0); n = fn.get(L,0)
        p = t/(t+f) if t+f else 0
        r = t/(t+n) if t+n else 0
        f1 = 2*p*r/(p+r) if p+r else 0
        out[L] = {"tp":t,"fp":f,"fn":n,"p":p,"r":r,"f1":f1,"support":t+n}
    return out


def cmp_block(name, m2, m3):
    print(f"\n=== {name} ===")
    for k in ("precision","recall","f1"):
        v2 = m2.get(k, 0); v3 = m3.get(k, 0)
        d = v3 - v2
        sign = "+" if d >= 0 else ""
        print(f"  {k:10s}  v2b={v2:.4f}  v3={v3:.4f}  Δ={sign}{d:.4f}")


# Top-level char metrics
v2b_char = load_char_metrics(V2B)
v3_char  = load_char_metrics(V3)
cmp_block("char-level typed (positive 1000)", v2b_char["typed"], v3_char["typed"])
cmp_block("char-level untyped (positive 1000)", v2b_char["untyped"], v3_char["untyped"])

# OPF detection / span (from positive_metrics)
m2 = json.load(open(f"{V2B}/positive_metrics.json"))
m3 = json.load(open(f"{V3}/positive_metrics.json"))
for k in ("detection","span"):
    if k in m2 and k in m3:
        cmp_block(f"OPF {k} (positive 1000)", m2[k], m3[k])

# Hardneg
hn2 = json.load(open(f"{V2B}/hardneg_metrics.json"))
hn3 = json.load(open(f"{V3}/hardneg_metrics.json"))
print("\n=== hardneg (4011 rows) ===")
for k in ("token_accuracy", "loss", "examples"):
    v2 = hn2.get("summary", {}).get(k)
    v3 = hn3.get("summary", {}).get(k)
    print(f"  {k:18s}  v2b={v2}  v3={v3}")

# Trap
tr2 = json.load(open(f"{V2B}/trap_metrics.json"))
tr3 = json.load(open(f"{V3}/trap_metrics.json"))
print("\n=== trap (100 rows) ===")
for k in ("token_accuracy", "loss"):
    v2 = tr2.get("summary", {}).get(k)
    v3 = tr3.get("summary", {}).get(k)
    print(f"  {k:18s}  v2b={v2}  v3={v3}")

# Per-label deltas focusing on weak classes
pl2 = per_label(V2B)
pl3 = per_label(V3)
weak_or_focus = ['SOCIAL_MEDIA_ID','SOCIO_ECONOMIC_STATUS','DEVICE_ID','PASSPORT_EXPIRY',
                 'AUDIO_INFORMATION','RACIAL_ETHNIC_ORIGIN','WEBSITE_HISTORY','FACIAL_RECOGNITION',
                 'CRIMINAL_RECORDS','NEXT_OF_KIN','FINGERPRINT','CAMERA_FOOTAGE_AUDIO',
                 'PRONOUN','SCHOLARSHIP','SALARY','SANCTIONS','MEDICARE_EXPIRY','PASSPORT_START_DATE',
                 'MEDICAL_INFORMATION','EMPLOYMENT_INFORMATION','SALARY_WAGE_EXPECTATION',
                 'ABORIGINALITY','PERSONAL_DEBT','SOCIAL_MEDIA_HISTORY','SOCIAL_MEDIA_ACCOUNT',
                 'AU_TFN','STUDENT_ID','PERSONNEL_NUMBER','CENTRELINK_REFERENCE_NUMBER']

print("\n=== per-label F1 changes (focus on weak classes) ===")
print(f"{'label':32s}  {'support':>7s}  {'v2b F1':>7s}  {'v3 F1':>7s}  {'delta':>7s}  {'note':12s}")
gains = []
for L in weak_or_focus:
    a = pl2.get(L); b = pl3.get(L)
    if not a or not b: continue
    d = b['f1'] - a['f1']
    gains.append((L, a, b, d))

# also include all classes sorted by absolute delta
all_labels = sorted(set(pl2) | set(pl3))
all_deltas = []
for L in all_labels:
    a = pl2.get(L, {"f1":0,"support":0})
    b = pl3.get(L, {"f1":0,"support":0})
    sup = max(a.get("support",0), b.get("support",0))
    if sup < 5: continue
    all_deltas.append((L, a, b, b.get("f1",0)-a.get("f1",0)))

for L, a, b, d in gains:
    note = "🔥 huge" if d >= 0.30 else ("✅ good" if d >= 0.10 else ("→ flat" if abs(d)<0.05 else ("⚠ regress" if d<0 else "↑ minor")))
    print(f"  {L:32s}  {a.get('support',0):7d}  {a['f1']:.4f}  {b['f1']:.4f}  {d:+.4f}  {note}")

# Top biggest gains and biggest regressions across ALL labels
print("\n=== top 10 biggest F1 gains (all labels, support≥5) ===")
all_deltas.sort(key=lambda x: -x[3])
for L, a, b, d in all_deltas[:10]:
    print(f"  {L:32s}  v2b={a['f1']:.4f} → v3={b['f1']:.4f}  Δ=+{d:.4f}  (sup={max(a.get('support',0),b.get('support',0))})")

print("\n=== top 10 biggest F1 regressions ===")
for L, a, b, d in all_deltas[-10:]:
    print(f"  {L:32s}  v2b={a['f1']:.4f} → v3={b['f1']:.4f}  Δ={d:+.4f}  (sup={max(a.get('support',0),b.get('support',0))})")
