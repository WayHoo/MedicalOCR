# coding=utf-8


def get_head_words_dict():
    """
    获取化验栏标题关键词dict
    :return: {word: sort_weight}，例如{"序号": 1, "项目名称": 1, "结果": 2}
    """
    head_words = {}
    with open("./doc/dict/test_sheet_head_dict.txt", "r", encoding="utf-8") as f:
        words = f.read().splitlines()
        sort_weight = 0
        for word in words:
            if word.startswith('#'):
                sort_weight = int(word[-1])
                continue
            head_words[word] = sort_weight
    return head_words


def get_key_words_set():
    """
    获取化验单中常出现的关键词集合
    :return: 关键词集合，如 {"项目名称", "姓名", "科室"}
    """
    key_words = set()
    with open("./doc/dict/test_sheet_kv_dict.txt", "r", encoding="utf-8") as f:
        words = f.read().splitlines()
        for word in words:
            key_words.add(word)
    return key_words
