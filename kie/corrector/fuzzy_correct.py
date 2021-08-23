"""
Reference: https://fuzzychinese.zenan-wang.com/
GitHub: https://github.com/znwang25/fuzzychinese
"""
import os
import logging
import pandas as pd
from fuzzychinese import FuzzyChineseMatch
default_logger = logging.getLogger(__name__)


class FuzzyCorrector(object):
    _dir = os.path.dirname
    _default_dict_file_path = os.path.join(_dir(_dir(_dir(__file__))),
                                           "doc/dict/medical_test_items.txt")

    def __init__(self, dict_file_path=None):
        if dict_file_path:
            self._dict_file_path = dict_file_path
        else:
            self._dict_file_path = self._default_dict_file_path
        self._read_dictionary()
        self.fcm = FuzzyChineseMatch(ngram_range=(3, 3), analyzer='stroke')
        self.fcm.fit(self._dictionary)

    def _read_dictionary(self):
        words = set()
        default_logger.debug('Reading test sheet dictionary ...')
        with open(self._dict_file_path, encoding="UTF-8") as f:
            for line in f:
                words.add(line.strip())
        self._dictionary = pd.Series(list(words))

    def get_top_candidates(self, words, n=1):
        if not isinstance(words, list):
            raise Exception('The param word must be type of list.')
        if not isinstance(n, int):
            raise Exception('The param n must be type of int.')
        if n <= 0:
            raise Exception('The param n must be positive integer.')
        top_n_similar = self.fcm.transform(words, n=n)
        top_n_score = self.fcm.get_similarity_score()
        ret = []
        for i in range(0, len(top_n_score)):
            zp = zip(top_n_similar[i], top_n_score[i])
            ret.append(list(zp))
        return ret


if __name__ == "__main__":
    fc = FuzzyCorrector()
    raw_words = ["申该细胞绝对值", "林巴细胞数", "载脂蛋白A", "钾",
                 "口嗜碱性粒细胞", "白细胞", "中形粒细胞数"]
    candi = fc.get_top_candidates(raw_words, 1)
    print(candi)
