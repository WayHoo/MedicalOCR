# coding=utf-8
import json


with open("./doc/dict/test_sheet_head.json", "r", encoding="utf-8") as f:
    print("start load test_sheet_head.json ......")
    GL_HEAD_WORDS = {}
    head_words = json.load(f)
    for item in head_words:
        for word in item["words"]:
            GL_HEAD_WORDS[word] = (item["id"], item["sort_weight"])


with open("./doc/dict/test_sheet_key_words.json", "r", encoding="utf-8") as f:
    print("start load test_sheet_key_words.json ......")
    key_words = json.load(f)
    GL_KEY_WORDS = set(key_words)


def is_cfg_head_word(word):
    """
    检查给定词语是否在配置的表头关键词中
    :param word: 字符串
    :return: True or False
    """
    return word in GL_HEAD_WORDS


def get_sort_weight(word):
    """
    获取表头词语的排序权重
    :param word: 字符串
    :return: 整型的排序权重
    """
    if word in GL_HEAD_WORDS:
        return GL_HEAD_WORDS[word][1]
    return 0


def is_cfg_key_word(word):
    """
    判断给定词语是否在配置的关键词中
    :param word: 字符串
    :return: True or False
    """
    return word in GL_KEY_WORDS


if __name__ == "__main__":
    print(GL_HEAD_WORDS)
    print(is_cfg_head_word("项目全称"))
    print(GL_KEY_WORDS)
