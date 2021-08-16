# coding=utf-8
from utils.item_match import match_item


def single_correct(word, category_id=1):
    pass


def correct_unit(word):
    value, span = match_item(word, category="unit")
    if value is not None:
        return value
    return word


def correct_number(word):
    pass