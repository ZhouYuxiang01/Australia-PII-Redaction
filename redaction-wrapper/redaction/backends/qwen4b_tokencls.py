"""Qwen4B Token Classifier backend v4.
Pipeline: decode -> crossline split -> entity tighten -> email rescue
         -> evidence gate -> negative context -> BSB rescue -> label normalize.
"""
from __future__ import annotations
import json, os, re, sys, threading
from pathlib import Path
from typing import Any
import torch, torch.nn.functional as F
from ..core.normalize import normalize_text
from ..core.span import Span
from .base import RedactionBackend

VALID_NEXT = {"B":{"I","E"},"I":{"I","E"},"E":{"B","S","O"},"S":{"B","S","O"},"O":{"B","S","O"}}
NEG_CTX = ("not a phone","not a student","not an id","not a pii","system-generated","system generated","permit ref","fake","sample","test token","placeholder","reference code","reference number","ticket","ticket id","invoice","room:","not staff","no pii","dummy","example email","demo","sandbox","sample email")
EMAIL_RGX = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", re.ASCII)
PHONE_RGX = re.compile(r"\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}")
CARD_RGX = re.compile(r"\b\d[\d\s\-]{12,22}\b")
SID_RGX = re.compile(r"SID#?\s*\d{6,12}", re.IGNORECASE)
DOB_RGX = re.compile(r"\b\d{1,2}[/\-. ]\d{1,2}[/\-. ]\d{2,4}\b")
PLATE_RGX = re.compile(r"\b[A-Z]{2,3}\s*[A-Z0-9]{2,6}\b")
BSB_RGX = re.compile(r"(?:BSB\s*:?\s*|bsb\s*:?\s*)(\d{3}[-\s]?\d{3})\b", re.IGNORECASE)
ACCT_RGX = re.compile(r"\b(?:account|acct)\s+(?:number\s+|no\.?\s+|#\s*)?(\d[\d\s]{5,18})", re.IGNORECASE)
SCHEMA_MAP = {"EMAIL_ADDRESS":"EMAIL","MOBILE":"PHONE","HOME_PHONE":"PHONE","WORK_PHONE":"PHONE","DRIVERS_LICENCE":"AU_DRIVERS_LICENCE","PASSPORT_NUMBER":"AU_PASSPORT","BANK_ACCOUNT_NUMBER":"AU_BANK_ACCOUNT","NUMBER_PLATE":"VEHICLE_ID","VEHICLE_REGO":"VEHICLE_ID"}
EXPANDABLE = {"EMAIL","EMAIL_ADDRESS","WORK_EMAIL","PHONE","MOBILE","HOME_PHONE","WORK_PHONE","PAYMENT_CARD_NUMBER","CREDIT_CARD_EXPIRY","AU_BANK_ACCOUNT","BANK_ACCOUNT_NUMBER"}

def _resolve_root(p=""):
    e=os.environ.get("REDACTION_PII_PROJECT_ROOT")
    if e: return Path(e)
    if p: return Path(p)
    for c in (Path(__file__).resolve().parents[3]/"pii_training_prep_v3_2", Path(__file__).resolve().parents[2]/"pii_training_prep_v3_2"):
        if c.is_dir(): return c
    raise FileNotFoundError("Cannot find pii_training_prep_v3_2")

def _neg_ctx(text,s,e):
    return any(p in text[max(0,s-80):e+20].lower() for p in NEG_CTX)

def _expand(text,s,e,t):
    if t in ("EMAIL","EMAIL_ADDRESS","WORK_EMAIL"):
        for m in EMAIL_RGX.finditer(text):
            if m.start()<=s<m.end() or m.start()<=e<=m.end(): return m.start(),m.end()
    if t in ("PHONE","MOBILE","HOME_PHONE","WORK_PHONE"):
        for m in PHONE_RGX.finditer(text):
            if m.start()<=s<m.end() or m.start()<=e<=m.end(): return m.start(),m.end()
    if t=="PAYMENT_CARD_NUMBER":
        for m in CARD_RGX.finditer(text):
            if m.start()<=s<m.end() or m.start()<=e<=m.end(): return m.start(),m.end()
    if t in ("AU_BANK_ACCOUNT","BANK_ACCOUNT_NUMBER"):
        for m in re.finditer(r"\b\d[\d\s\-]{5,18}\b",text):
            if m.start()<=s<m.end() or m.start()<=e<=m.end(): return m.start(),m.end()
    return s,e

def _agg_entity(logits,ti,id2l,k=5):
    if not ti: return 0.0,0.0,[]
    p=F.softmax(logits,dim=-1); mp=p[ti].mean(dim=0); es={}
    for lid in range(1,logits.shape[1]):
        ln=id2l.get(str(lid),""); tag=ln[:2] if len(ln)>=2 else ""
        if tag in ("B-","I-","E-","S-"): entity=ln[2:]; es[entity]=es.get(entity,0)+mp[lid].item()
    se=sorted(es.items(),key=lambda x:x[1],reverse=True)
    tk=[(e,round(s,6)) for e,s in se[:k]]; t1=tk[0][1] if tk else 0; t3=sum(s for _,s in tk[:3])
    return round(t1,6),round(t3,6),tk

def _split_newlines(spans,text):
    out=[]
    for sp in spans:
        val=text[sp["start"]:sp["end"]]
        if "\n" not in val: out.append(sp); continue
        off=sp["start"]
        for line in val.split("\n"):
            if not line.strip(): off+=len(line)+1; continue
            ls=off+(len(line)-len(line.lstrip())); le=off+len(line.rstrip())
            if le>ls: out.append({**sp,"start":ls,"end":le,"num_tokens":max(1,sp["num_tokens"]//max(1,len(val.split(chr(10)))-1))})
            off+=len(line)+1
    out.sort(key=lambda s:s["start"])
    return out

def _tighten(sp,text):
    t=sp["type"]; val=text[sp["start"]:sp["end"]]
    if t=="STUDENT_ID":
        m=SID_RGX.search(val)
        if m: return {**sp,"start":sp["start"]+m.start(),"end":sp["start"]+m.end()}
        m=re.search(r"\d{6,12}",val)
        if m: return {**sp,"start":sp["start"]+m.start(),"end":sp["start"]+m.end()}
    if t=="DATE_OF_BIRTH":
        m=DOB_RGX.search(val)
        if m: return {**sp,"start":sp["start"]+m.start(),"end":sp["start"]+m.end()}
    if t=="VEHICLE_ID":
        m=PLATE_RGX.search(val)
        if m: return {**sp,"start":sp["start"]+m.start(),"end":sp["start"]+m.end()}
    if t in ("NEXT_OF_KIN","PHONE","MOBILE","HOME_PHONE","WORK_PHONE"):
        if "\n" in val: return None
    if t=="PERSON" and val.strip().startswith("@"): return None
    return sp

def _rescue_emails(text,existing):
    seen={(s["start"],s["end"]) for s in existing}; out=[]
    for m in EMAIL_RGX.finditer(text):
        s,e=m.start(),m.end()
        if (s,e) not in seen: out.append({"start":s,"end":e,"type":"EMAIL","confidence":1.0,"num_tokens":1,"token_indices":[]})
    return out

def _rescue_bsb(text,existing):
    out=[]; seen={(s.start,s.end) for s in existing}
    for m in BSB_RGX.finditer(text):
        v=m.group(1).rstrip(); s,e=m.start(1),m.end(1)
        if _neg_ctx(text,s,e): continue
        if "\n" in text[max(0,s-30):e+30]: continue
        if (s,e) not in seen:
            sp=Span(start=s,end=e,type="AU_BANK_ACCOUNT",value=v,confidence=1.0,decision="AUTO_REDACT",replacement="[AU_BANK_ACCOUNT]",source="rule")
            sp.top_type="AU_BANK_ACCOUNT"; sp.top_probability=1.0; sp.top1_prob=1.0; sp.top3_sum=1.0
            sp.type_distribution_topk=[["AU_BANK_ACCOUNT",1.0]]; sp.decision_reason="bsb_regex_rescue"; sp.policy_version="v4"
            out.append(sp); seen.add((s,e))
    for m in ACCT_RGX.finditer(text):
        v=m.group(1).strip(); s,e=m.start(1),m.end(1)
        if _neg_ctx(text,s,e): continue
        if "\n" in text[max(0,s-5):e+5]: continue
        if not re.match(r"^[\d\s\-]+$",v): continue
        if (s,e) not in seen:
            sp=Span(start=s,end=e,type="AU_BANK_ACCOUNT",value=v,confidence=1.0,decision="AUTO_REDACT",replacement="[AU_BANK_ACCOUNT]",source="rule")
            sp.top_type="AU_BANK_ACCOUNT"; sp.top_probability=1.0; sp.top1_prob=1.0; sp.top3_sum=1.0
            sp.type_distribution_topk=[["AU_BANK_ACCOUNT",1.0]]; sp.decision_reason="account_regex_rescue"; sp.policy_version="v4"
            out.append(sp); seen.add((s,e))
    return out


class Qwen4BTokenClsBackend(RedactionBackend):
    def __init__(self,*,name="qwen4b-tokencls",model_version="qwen4b-tokencls-v1",supported_types=None,
                 model_path="/home/admin/model/Qwen3.5-4B-Base",checkpoint_path="",token_label_to_id_path="",
                 id_to_token_label_path="",max_seq_len=4096,dtype="bf16",device="cuda",output_top_k=5,pii_project_root=""):
        self._name=name; self._model_version=model_version; self._model_path=model_path
        self._max_seq_len=int(max_seq_len); self._dtype=dtype; self._device=device; self._output_top_k=int(output_top_k)
        pr=_resolve_root(pii_project_root); self._pii_root=pr
        self._ckpt=Path(checkpoint_path) if checkpoint_path else pr/"runs"/"qwen4b_tokencls_head_only"/"best_head.pt"
        self._tl2i=Path(token_label_to_id_path) if token_label_to_id_path else pr/"pii_schema"/"token_label_to_id_317.json"
        self._i2tl=Path(id_to_token_label_path) if id_to_token_label_path else pr/"pii_schema"/"id_to_token_label_317.json"
        if supported_types: self._st=list(supported_types)
        else: self._st=list(json.loads((pr/"pii_schema"/"canonical_labels_79.json").read_text()))
        self._lock=threading.Lock(); self._loaded=False; self._model=None; self._tok=None
        self._l2i={}; self._i2l={}; self._hs=2560

    @property
    def name(self): return self._name
    @property
    def model_version(self): return self._model_version
    @property
    def supported_types(self): return self._st

    def load(self):
        if self._loaded: return
        with self._lock:
            if self._loaded: return
            self._l2i=json.loads(self._tl2i.read_text())["label_to_id"]
            self._i2l=json.loads(self._i2tl.read_text())["id_to_label"]
            sys.path.insert(0,str(self._pii_root/"src"/"pii_prep"))
            from qwen4b_tokencls_model import load_model
            bf=self._dtype=="bf16" and torch.cuda.is_bf16_supported()
            dev=torch.device(self._device if torch.cuda.is_available() else "cpu")
            m,tk,hs=load_model(self._model_path,num_labels=317,freeze_backbone=True,device=dev,use_bf16=bf)
            self._model=m; self._tok=tk; self._hs=hs
            ck=torch.load(str(self._ckpt),map_location="cpu",weights_only=False)
            self._model.classifier.load_state_dict(ck["head_state_dict"]); self._model.eval(); self._loaded=True

    def _tag(self,lid):
        if lid==0: return "O"
        ln=self._i2l.get(str(lid),"O")
        if ln.startswith("B-"): return "B"
        if ln.startswith("I-"): return "I"
        if ln.startswith("E-"): return "E"
        if ln.startswith("S-"): return "S"
        return "O"

    def _entity(self,lid):
        if lid<=0: return "O"
        ln=self._i2l.get(str(lid),"O")
        for p in ("B-","I-","E-","S-"):
            if ln.startswith(p): return ln[2:]
        return ln

    def _decode(self,logits):
        pr=F.softmax(logits,dim=-1); n=logits.shape[0]; preds=[0]*n
        for i in range(n):
            vt={"B","S","O"} if i==0 else VALID_NEXT.get(self._tag(preds[i-1]),{"B","S","O"})
            vi=[l for l in range(logits.shape[1]) if self._tag(l) in vt]
            if not vi: vi=[0]
            bi=vi[0]; bs=-float("inf")
            for l in vi:
                sc=pr[i,l].item()
                if sc>bs: bs=sc; bi=l
            preds[i]=bi
        return preds

    def _decode_spans(self,logits,off):
        preds=self._decode(logits); pr=F.softmax(logits,dim=-1); cf=pr[torch.arange(len(preds)),preds].tolist()
        n=len(preds); spans=[]; i=0
        while i<n:
            l=preds[i]; tag=self._tag(l)
            if tag=="O": i+=1; continue
            if tag=="S":
                e=self._entity(l)
                if e!="O" and i<len(off):
                    s,e2=off[i]
                    if s<e2: spans.append({"s":s,"e":e2,"t":e,"c":round(cf[i],6),"nt":1,"ti":[i]})
                i+=1; continue
            if tag=="B":
                e=self._entity(l); si=i; ei=i+1; sc=[cf[i]]; ti=[i]; j=i+1
                while j<n:
                    nl=preds[j]; nt=self._tag(nl)
                    if nt in("I","E"): ei=j+1; sc.append(cf[j]); ti.append(j)
                    if nt=="E": j+=1; break
                    if nt=="I": j+=1
                    else: break
                if si<len(off) and ei<=len(off):
                    s,e2=off[si][0],off[ei-1][1]
                    if s<e2 and e!="O": spans.append({"s":s,"e":e2,"t":e,"c":round(sum(sc)/len(sc),6),"nt":ei-si,"ti":ti})
                i=max(ei,i+1); continue
            i+=1
        spans=[s for s in spans if s["s"]<s["e"] and s["c"]>0]
        spans.sort(key=lambda s:(-s["c"],-(s["e"]-s["s"])))
        res=[]
        for sp in spans:
            if not any(sp["s"]<r["e"] and sp["e"]>r["s"] for r in res): res.append(sp)
        res.sort(key=lambda s:s["s"])
        return res

    def detect_spans(self,text):
        self.load(); text=normalize_text(text)
        enc=self._tok(text,truncation=True,max_length=self._max_seq_len,return_offsets_mapping=True,return_attention_mask=True,return_tensors="pt")
        ids=enc["input_ids"].to(self._model.classifier.weight.device)
        mask=enc["attention_mask"].to(self._model.classifier.weight.device)
        off=enc["offset_mapping"][0].tolist()
        with torch.no_grad(): logits=self._model.predict(ids,mask)[0].cpu()
        vl=mask[0].sum().item(); logits=logits[:vl]; off=[(int(a),int(b)) for a,b in off[:vl]]
        decoded=self._decode_spans(logits,off)
        decoded=[{"start":d["s"],"end":d["e"],"type":d["t"],"confidence":d["c"],"num_tokens":d["nt"],"token_indices":d["ti"]} for d in decoded]
        decoded=_split_newlines(decoded,text)
        decoded=[_tighten(s,text) for s in decoded]
        decoded=[s for s in decoded if s is not None and s["start"]<s["end"] and not (s["type"]=="STUDENT_ID" and not re.search(r"\d",text[s["start"]:s["end"]]))]
        decoded.extend(_rescue_emails(text,decoded))
        spans=[]
        for ds in decoded:
            t0=ds["type"]; start=ds["start"]; end=ds["end"]; conf=ds["confidence"]; ti=ds.get("token_indices",[])
            if t0 in EXPANDABLE: start,end=_expand(text,start,end,t0)
            if start>=end: continue
            val=text[start:end]
            if val.endswith("\n"): val=val.rstrip("\n"); end=start+len(val)
            top1,top3,topk=_agg_entity(logits,ti,self._i2l,self._output_top_k)
            is_rescue = ds.get("confidence",0)>=0.99 and not ti
            if is_rescue: top1=conf; top3=conf; topk=[[t0,conf]]
            neg=False if is_rescue else _neg_ctx(text,start,end)
            if not is_rescue and top1<0.20 and top3<0.40: decision="REVIEW"; reason="low_pii_evidence"
            elif neg: decision="REVIEW"; reason="negative_context"
            else: decision="AUTO_REDACT"; reason="qwen4b_tokencls_bioes_decode"
            nt=SCHEMA_MAP.get(t0,t0)
            sp=Span(start=start,end=end,type=nt,value=val,confidence=conf,decision=decision,replacement=f"[{nt}]",source="model")
            sp.top_type=nt; sp.top_probability=top1; sp.top1_prob=top1; sp.top3_sum=top3
            sp.non_pii_prob=round(F.softmax(logits[ti],dim=-1)[:,0].mean().item(),6) if ti else 0.0
            sp.type_distribution_topk=[[t,p] for t,p in topk]
            sp.decision_reason=reason; sp.policy_version="v4"; sp.pii_evidence_passed=decision=="AUTO_REDACT"; sp.evidence_reason=reason
            spans.append(sp)
        rescued=_rescue_bsb(text,spans)
        for s in rescued: s.type=SCHEMA_MAP.get(s.type,s.type); s.top_type=s.type
        spans.extend(rescued)
        return spans,{"warnings":[],"raw_offset_mapping_applied":False,"decoded":len(decoded),"rescued":len(rescued),"output":len(spans)}

    def info(self):
        b=super().info(); b.update(pipeline="qwen4b-tokencls-head",hidden_size=self._hs,max_seq_len=self._max_seq_len,
            checkpoint=str(self._ckpt),output_top_k=self._output_top_k,
            features=["entity_topk","boundary_expansion","crossline_split","entity_tighten","email_rescue","evidence_gate","negative_context","bsb_rescue","label_normalization"])
        return b
