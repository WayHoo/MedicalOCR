# coding=utf-8
import re


def match_item(source, category="unit"):
    if source == "":
        return None, None
    if category == "unit":
        pairs = [(r"[pP][eE][iI1l][uU]/m[1IlL]", "PEIU/mL"),
                 (r"10.?3/[1IlL]?",              "10^3/L"),
                 (r"10.?6/[1IlL]?",              "10^6/L"),
                 (r"10.?9/[1IlL]?",              "10^9/L"),
                 (r"10.?12/[1IlL]?",             "10^12/L"),
                 (r"mm[0oO][1IlL]/[1IlL]",       "mmol/L"),
                 (r"[uμ]m[0oO][1IlL]/[1IlL]",    "μmol/L"),
                 (r"m[0oO][1IlL]/[1IlL]",        "mol/L"),
                 (r"m[I1][uU]/m[1IlL]",          "mIU/mL"),
                 (r"[I1][uU]/m[1IlL]",           "IU/mL"),
                 (r"mg/d[1IlL]",                 "mg/dL"),
                 (r"ng/m[1IlL]",                 "ng/mL"),
                 (r"mg/[1IlL]",                  "mg/L"),
                 (r"[I1][uU]/[1IlL]",            "IU/L"),
                 (r"s/c[o0O]",                   "s/co"),
                 (r"[uU]/[1IlL]",                "U/L"),
                 (r"[g/[1IlL]]",                 "g/L"),
                 (r"[1IlL]/[1IlL]",              "L/L"),
                 (r"/[hH][pP]",                  "/HP"),
                 (r"f[1IlL]",                    "fL"),
                 (r"pg",                         "pg"),
                 ("%",                           "%")]
        for pair in pairs:
            pattern = re.compile(pair[0])
            m = re.search(pattern, source)
            if m is not None:
                return pair[1], m.span()
    elif category == "number":
        regs = [r"-?(\d*\.\d*|\d*)"]
        for reg in regs:
            pattern = re.compile(reg)
            m = re.search(pattern, source)
            if m is not None:
                reg_str = m.group()
                num_str = reg_str_to_int(reg_str)
                return num_str, m.span()
    return None, None


def reg_str_to_int(reg_str):
    try:
        n = float(reg_str)
        if reg_str.find('.') < 0:
            n = str(int(n))
        return n
    except:
        print("can't convert str to number, reg_str=%s" % reg_str)
    return reg_str


def item_seg(s):
    value, span = match_item(s, category="unit")
    if value is not None and value != "":
        print(value, span)
        if span[1] == len(s):
            return value, s[:span[0]], ""
        else:
            return value, s[:span[0]], s[span[1]:]
    return s, ""


if __name__ == "__main__":
    units = ["mmo1/Lmol/l", "umO1/l", "umol/L", "mmo1/L", "mo1/L", "mmol/L", "mg/dL", "10~12/L",
             "10~9/", "←4.00~10.0(10~9//", "3.50~5.501012/L"]
    match_units = []
    for candi in units:
        res = match_item(candi, category="unit")
        print(res)
    numbers = ["123.", "0.312", "0", "-12.1", "00012.3"]
    for num in numbers:
        res = match_item(num, category="number")
        print(res)
    item_seg("79.6486.0~100.(fl")
