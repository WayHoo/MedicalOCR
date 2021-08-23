# coding=utf-8
from utils.head_cfg import get_sort_weight
from utils.xlsx import write_excel_xlsx

__all__ = ["parse_sheet_to_excel"]


def parse_sheet_header(heads):
    word_set = set()
    for word in heads:
        if word not in word_set:
            word_set.add(word)
    visited, idx = {heads[0]}, 1
    while idx < len(heads):
        if heads[idx] in visited or get_sort_weight(heads[idx]) < get_sort_weight(heads[idx-1]):
            break
        idx += 1
    if idx == len(heads):
        return heads, 1
    part1, part2 = heads[:idx], heads[idx:]
    part1_set, part2_set = set(part1), set(part2)
    merged = []
    idx1, idx2 = 0, 0
    while idx1 < len(part1) and idx2 < len(part2):
        a, b = part1[idx1], part2[idx2]
        if a == b:
            merged.append(a)
            idx1 += 1
            idx2 += 1
        elif b in part1_set:
            merged.append(a)
            idx1 += 1
        elif a in part2_set:
            merged.append(b)
            idx2 += 1
        elif get_sort_weight(a) <= get_sort_weight(b):
            merged.append(a)
            idx1 += 1
        else:
            merged.append(b)
            idx2 += 1
    while idx1 < len(part1):
        merged.append(part1[idx1])
        idx1 += 1
    while idx2 < len(part2):
        merged.append(part2[idx2])
        idx2 += 1
    return merged, 2


def parse_sheet_data(heads, body_lines):
    """
    解析化验单内容，如果为双栏，解析为单栏
    :param heads: 化验单表头
    :param body_lines: 化验单化验结果行
    :return: 可写入 excel 的化验单内容
    """
    merged_heads, sheet_cnt = parse_sheet_header(heads)
    data = [merged_heads]
    sheets = [[], []]
    for line in body_lines:
        idx = 0
        content, i = [[], []], 0
        for meta in line:
            attrs = meta["attrs"] if "attrs" in meta else []
            for attr_idx, attr in enumerate(attrs):
                while idx < len(merged_heads):
                    idx += 1
                    if attr == merged_heads[idx-1]:
                        content[i].append(meta["corrected"][attr_idx])
                        break
                    else:
                        content[i].append("")
                if idx == len(merged_heads):
                    idx = 0
                    i += 1
        for j in range(0, len(sheets)):
            if len(content[j]) > 0:
                sheets[j].append(content[j])
    for sheet in sheets:
        data.extend(sheet)
    return data


def parse_sheet_to_excel(heads, body_lines, file_name,
                         file_path="./output/inference_results/test_sheets/",
                         sheet_name="化验单"):
    if len(heads) == 0 or len(body_lines) == 0 or file_name == "":
        print("illegal params to write_excel_sheet!!!")
        return
    data = parse_sheet_data(heads, body_lines)
    if len(data) == 0:
        print("no valid data to write excel!!!")
        return
    write_excel_xlsx(file_path, file_name, sheet_name, data)
    return


if __name__ == "__main__":
    origin_heads = ["项目名称", "英文缩写", "结果", "单位", "参考区间", "NO", "英文缩写", "单位", "参考区间"]
    body = [[{"text": "钾离子(K)", "attrs": ["项目名称", "英文缩写"]},
             {"text": "4.20", "attrs": ["结果"]}, {"text": "mmol/L", "attrs": ["单位"]},
             {"text": "3.50-5.30", "attrs": ["参考区间"]}, {"text": "3", "attrs": ["NO"]},
             {"text": "TP", "attrs": ["英文缩写"]}, {"text": "μmol/L", "attrs": ["单位"]},
             {"text": "0.0-6.8", "attrs": ["参考区间"]}],
            [{"text": "钠离子", "attrs": ["项目名称"]}, {"text": "Na", "attrs": ["英文缩写"]},
             {"text": "140", "attrs": ["结果"]}, {"text": "mmol/L", "attrs": ["单位"]},
             {"text": "137-147", "attrs": ["参考区间"]}, {"text": "4", "attrs": ["NO"]},
             {"text": "UA", "attrs": ["英文缩写"]}, {"text": "U/L", "attrs": ["单位"]},
             {"text": "3.0-6.8", "attrs": ["参考区间"]}]]
    data = parse_sheet_data(origin_heads, body)
    for d in data:
        print(d)
