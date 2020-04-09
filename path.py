# -*- coding: utf-8 -*
import sys
import os

DATA_PATH = os.path.join(sys.path[0], 'data', 'input')
MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')

import re


def get_word_list(s1):
    # 把句子按字分开，中文按字分，英文按单词，数字按空格
    # 我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
    # reg = re.compile('[^a-zA-Z0-9\u4e00-\u9fa5\u3002-]|[.,!?;]')
    #
    # return reg.split(s1.lower())

    return re.findall(r"[\w']+|[,!?;]", s1.lower())


print(get_word_list('SELECT DISTINCT t1.individual_last_name FROM individuals AS t1 JOIN organization_contact_individuals AS t2 ON t1.individual_id  =  t2.individual_id'))
print(get_word_list('how many different advisors are listed?'))
print(get_word_list('how many different advisors are listed?'))
print(get_word_list('how many different advisors are listed?'))
print(get_word_list('how many different advisors are listed?'))
print(get_word_list('how many different advisors are listed?'))
