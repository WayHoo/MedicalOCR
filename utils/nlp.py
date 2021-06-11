import jieba


def jieba_seg(text):
    """
    结巴分词，自定义词典，精确模式
    :param text: 待分词的文本字符串
    :return: 词语列表
    """
    jieba.load_userdict("./doc/dict/test_sheet_seg_dict.txt")
    seg_list = jieba.cut(text, cut_all=False)  # 精确模式
    words = []
    for seg in seg_list:
        words.append(seg)
    return words


if __name__ == "__main__":
    text_list = ["项目结果生物参考区间提示单位NO检验项目检验结果参考范围单位No",
                 "序号：0024", "检验项目名称", "性别：女标本种类：血清送检医生："]
    for text in text_list:
        print(jieba_seg(text))
