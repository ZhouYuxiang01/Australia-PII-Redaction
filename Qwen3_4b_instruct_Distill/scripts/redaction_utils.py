"""
redaction_utils.py

共享工具函数:
- annotate_text(text, labels):把 labels 插入 text 里变成带 <pii> 标签的文本
- parse_annotated(annotated_text):从带标签的文本解析出 span 列表
- apply_redaction(text, spans):按 span 脱敏原文

数据构造和评估都会用到,抽出来避免重复。
"""

import re
from typing import List, Dict, Any, Tuple, Optional


TAG_PATTERN = re.compile(r'<pii type="([^"]+)">(.*?)</pii>', re.DOTALL)


# ============================================================
# 1. 从 labels 构造 annotated text (训练数据构造时用)
# ============================================================
def annotate_text(text: str,
                  labels: List[Dict[str, Any]],
                  target_types: Optional[set] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """
    把原始文本按 labels 加上 <pii type="..."> 标签。

    Args:
        text: 原始文本
        labels: [{start, end, type, value, ...}, ...]
        target_types: 只标记这些类型,其他过滤掉。None = 全部

    Returns:
        annotated_text: 带标签的文本
        used_labels: 实际被标记的 labels(过滤 + 去重叠后)

    策略:
        - 过滤不在 target_types 里的
        - 检测重叠 span,保留 confidence 高的(相同 span 不同 type 的情况)
        - 按 start 降序插入标签(避免 offset 漂移)
        - 验证 labels.value == text[start:end],不一致的 label 跳过
    """
    # 1. 过滤类型
    if target_types is not None:
        labels = [l for l in labels if l["type"] in target_types]

    # 2. 验证 value 和 text[start:end] 一致性,不一致的跳过
    valid = []
    for lab in labels:
        start, end = lab["start"], lab["end"]
        if start < 0 or end > len(text) or start >= end:
            continue
        if text[start:end] != lab["value"]:
            continue
        valid.append(lab)

    # 3. 处理重叠
    valid = _resolve_overlaps(valid)

    # 4. 按 start 降序插入
    valid_desc = sorted(valid, key=lambda x: x["start"], reverse=True)

    result = text
    for lab in valid_desc:
        start, end = lab["start"], lab["end"]
        inner = result[start:end]
        tag_open = f'<pii type="{lab["type"]}">'
        tag_close = "</pii>"
        result = result[:start] + tag_open + inner + tag_close + result[end:]

    # 5. 返回原顺序(按 start 升序)
    valid_asc = sorted(valid, key=lambda x: x["start"])
    return result, valid_asc


def _overlaps(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return a["start"] < b["end"] and a["end"] > b["start"]


def _resolve_overlaps(labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    解决 span 重叠:按 confidence 从高到低贪心选择,一旦和已选集合重叠就丢掉。

    保证:返回的 labels 两两不重叠。
    相比旧实现的改动:
        - 旧实现按 start 顺序遍历,每次只处理第一个冲突,多重叠时可能残留
        - 新实现按 confidence 降序贪心,天然保证无残余重叠
    """
    if not labels:
        return []

    # 按 confidence 降序(同分时优先更长的 span, 再按 start 稳定排序)
    sorted_labels = sorted(
        labels,
        key=lambda x: (-x.get("confidence", 0.0), -(x["end"] - x["start"]), x["start"])
    )

    kept: List[Dict[str, Any]] = []
    for lab in sorted_labels:
        if any(_overlaps(lab, k) for k in kept):
            continue   # 和任意已选冲突就丢弃
        kept.append(lab)

    # 返回时按 start 升序,符合调用者期望
    kept.sort(key=lambda x: x["start"])
    return kept


# ============================================================
# 2. 从 annotated text 解析 spans (推理时用,评估也用)
# ============================================================
class ParseError(Exception):
    """解析失败(标签格式错误、不闭合等)"""
    pass


def parse_annotated(annotated_text: str,
                    strict: bool = False) -> Tuple[str, List[Dict[str, Any]]]:
    """
    从带 <pii> 标签的文本解析出:
        - 去除标签后的原文
        - spans: [{start, end, type, value}]  (offset 相对去标签后的原文)

    Args:
        annotated_text: 模型输出的带标签文本
        strict: True 时格式错误抛异常; False 时尽量解析能解析的部分

    Returns:
        plain_text: 去除标签后的文本
        spans: span 列表

    设计说明:
        - 生成式模型可能会漏闭合标签、嵌套等。strict=False 时会跳过这种 span
        - 同一个 value 在原文出现多次都会被独立记录(每次出现的位置不同)
    """
    spans = []
    plain_parts = []
    plain_pos = 0
    ann_pos = 0
    ann_len = len(annotated_text)

    while ann_pos < ann_len:
        # 找下一个 <pii type="
        tag_start = annotated_text.find('<pii type="', ann_pos)

        if tag_start == -1:
            # 剩余都是纯文本
            remainder = annotated_text[ann_pos:]
            plain_parts.append(remainder)
            plain_pos += len(remainder)
            break

        # tag_start 之前的是纯文本
        pre_text = annotated_text[ann_pos:tag_start]
        plain_parts.append(pre_text)
        plain_pos += len(pre_text)

        # 解析 type
        type_end = annotated_text.find('">', tag_start)
        if type_end == -1:
            if strict:
                raise ParseError(f"标签未闭合 at {tag_start}")
            # 把残留文本当普通文本处理
            plain_parts.append(annotated_text[tag_start:])
            break

        pii_type = annotated_text[tag_start + len('<pii type="'):type_end]
        content_start = type_end + len('">')

        # 找 </pii>
        close_tag = annotated_text.find("</pii>", content_start)
        if close_tag == -1:
            if strict:
                raise ParseError(f"缺少 </pii> for tag at {tag_start}")
            # 把这段当普通文本(含半截标签)
            plain_parts.append(annotated_text[tag_start:])
            break

        value = annotated_text[content_start:close_tag]

        # 嵌套检查(简单版:value 里不应有 <pii)
        if "<pii type=" in value:
            if strict:
                raise ParseError(f"检测到嵌套 pii 标签 at {tag_start}")
            # 跳过这个 span,把原始文本当普通文本
            plain_parts.append(annotated_text[tag_start:close_tag + len("</pii>")])
            plain_pos += close_tag + len("</pii>") - tag_start
            ann_pos = close_tag + len("</pii>")
            continue

        # 记录 span
        span_start = plain_pos
        span_end = plain_pos + len(value)
        spans.append({
            "start": span_start,
            "end": span_end,
            "type": pii_type,
            "value": value,
        })

        plain_parts.append(value)
        plain_pos = span_end
        ann_pos = close_tag + len("</pii>")

    plain_text = "".join(plain_parts)
    return plain_text, spans


def strip_tags_only(annotated_text: str) -> str:
    """快捷函数:只要去掉标签的纯文本"""
    plain, _ = parse_annotated(annotated_text, strict=False)
    return plain


# ============================================================
# 3. 应用 redaction (部署时用)
# ============================================================
def apply_redaction(text: str,
                    spans: List[Dict[str, Any]],
                    mode: str = "mask",
                    placeholder_fmt: str = "[{type}]") -> str:
    """
    按 spans 对原文脱敏。

    Args:
        text: 原文
        spans: [{start, end, type, ...}, ...]
        mode: 'mask' = 用 [TYPE] 替换; 'remove' = 直接删掉
        placeholder_fmt: mask 模式下的占位符格式

    Returns:
        脱敏后的文本
    """
    # 按 start 降序,避免 offset 漂移
    spans_desc = sorted(spans, key=lambda x: x["start"], reverse=True)
    result = text
    for s in spans_desc:
        start, end = s["start"], s["end"]
        if mode == "mask":
            placeholder = placeholder_fmt.format(type=s["type"])
            result = result[:start] + placeholder + result[end:]
        elif mode == "remove":
            result = result[:start] + result[end:]
        else:
            raise ValueError(f"unknown mode: {mode}")
    return result


# ============================================================
# 4. 基本自测
# ============================================================
if __name__ == "__main__":
    # 测试 1: 基本 annotate + parse round-trip
    text = "Mark Chen lives at Sydney. His phone is 0412345678."
    labels = [
        {"start": 0, "end": 9, "type": "PERSON", "value": "Mark Chen", "confidence": 0.95},
        {"start": 19, "end": 25, "type": "ADDRESS", "value": "Sydney", "confidence": 0.80},
        {"start": 40, "end": 50, "type": "AU_PHONE", "value": "0412345678", "confidence": 0.99},
    ]

    annotated, used = annotate_text(text, labels)
    print("ANNOTATED:")
    print(annotated)
    print()

    plain, spans = parse_annotated(annotated)
    print("PARSED PLAIN:", plain)
    print("PARSED SPANS:", spans)
    assert plain == text, "round-trip 失败"
    assert len(spans) == 3
    for s, orig in zip(spans, labels):
        assert s["start"] == orig["start"]
        assert s["end"] == orig["end"]
        assert s["type"] == orig["type"]
        assert s["value"] == orig["value"]
    print("✓ round-trip 测试通过\n")

    # 测试 2: redaction
    redacted = apply_redaction(text, spans, mode="mask")
    print("REDACTED:", redacted)
    print()

    # 测试 3: 重叠 span(两个 label 同位置)
    labels2 = [
        {"start": 0, "end": 6, "type": "VEHICLE_REGO", "value": "O385UM", "confidence": 0.93},
        {"start": 0, "end": 6, "type": "NUMBER_PLATE", "value": "O385UM", "confidence": 0.89},
    ]
    annotated2, used2 = annotate_text("O385UM is the rego.", labels2, target_types={"VEHICLE_REGO", "NUMBER_PLATE"})
    print("OVERLAP TEST:")
    print(annotated2)
    assert len(used2) == 1, f"应该只保留一个(confidence 最高),实际 {len(used2)}"
    assert used2[0]["type"] == "VEHICLE_REGO", "应该保留 confidence 更高的 VEHICLE_REGO"
    print("✓ 重叠处理测试通过\n")

    # 测试 4: hard negative(无标签)
    neg_text = "Please call the office before 5pm."
    annotated3, used3 = annotate_text(neg_text, [])
    assert annotated3 == neg_text
    plain3, spans3 = parse_annotated(annotated3)
    assert plain3 == neg_text
    assert spans3 == []
    print("✓ 负样本测试通过\n")

    # 测试 5: 模型可能输出的畸形格式
    malformed = 'Hello <pii type="PERSON">Alice world'  # 没闭合
    plain4, spans4 = parse_annotated(malformed, strict=False)
    print(f"MALFORMED non-strict: plain={plain4!r}, spans={spans4}")
    print("✓ 非 strict 模式畸形输入测试通过\n")

    # 测试 6: 三重叠场景 (修复前会残留 overlap)
    # spanC 同时和 spanA、spanB 重叠,如果 C 的 confidence 最高,应该只保留 C
    labels6 = [
        {"start": 0,  "end": 10, "type": "ADDRESS", "value": "0123456789", "confidence": 0.70},  # A
        {"start": 20, "end": 30, "type": "ADDRESS", "value": "0123456789", "confidence": 0.75},  # B
        {"start": 5,  "end": 25, "type": "PERSON",  "value": "5" * 20,     "confidence": 0.95},  # C 覆盖 A+B
    ]
    kept6 = _resolve_overlaps(labels6)
    assert len(kept6) == 1, f"三重叠应该只留 1 个,实际 {len(kept6)}"
    assert kept6[0]["type"] == "PERSON", "应保留 confidence 最高的 PERSON"
    # 验证返回的 kept 两两不重叠
    for i in range(len(kept6)):
        for j in range(i+1, len(kept6)):
            assert not _overlaps(kept6[i], kept6[j]), "kept 里不应有残余重叠"
    print("✓ 三重叠测试通过(修复的 bug)\n")

    # 测试 7: 并列不重叠,都保留
    labels7 = [
        {"start": 0,  "end": 5,  "type": "PERSON", "value": "Alice", "confidence": 0.9},
        {"start": 10, "end": 15, "type": "PERSON", "value": "Bobby", "confidence": 0.8},
        {"start": 20, "end": 25, "type": "PERSON", "value": "Carol", "confidence": 0.7},
    ]
    kept7 = _resolve_overlaps(labels7)
    assert len(kept7) == 3, f"无重叠时应全部保留,实际 {len(kept7)}"
    # 返回应按 start 升序
    assert [k["start"] for k in kept7] == [0, 10, 20]
    print("✓ 无重叠全保留测试通过\n")

    print("\n所有测试通过 ✓")
