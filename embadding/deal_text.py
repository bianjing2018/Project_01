import jieba
import re


def cut(sentence):
    pattern = re.compile('[\w+]')
    sentence = ''.join(re.findall(pattern, sentence))
    return jieba.lcut(sentence)

