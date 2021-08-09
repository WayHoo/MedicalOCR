import jieba


def jieba_seg(text):
    """
    结巴分词，自定义词典，精确模式
    :param text: 待分词的文本字符串
    :return: 词语列表
    """
    seg_list = jieba.cut(text, cut_all=False)  # 精确模式
    words = []
    for seg in seg_list:
        words.append(seg)
    return words


if __name__ == "__main__":
    text_list = ["项目结果生物参考区间提示单位NO检验项目检验结果参考范围单位No",
                 "序号：0024", "检验项目名称", "性别：女标本种类：血清送检医生：",
                 "序号项目代号项目名称结果单位参考值",
                 "1 钾离了（K） 4.20 mmo1/L 3.50-5.30 16直接胆红素（DBIL） 3.0 umol/L 0.0-6.8",
                 "采集：2018-03-0714:32", "BUN 尿素氮 7.1 2.8-7.2 mmol/LB-MGB微球蛋白 1.15 0-3 mg/L"]
    for text in text_list:
        print(jieba_seg(text))
