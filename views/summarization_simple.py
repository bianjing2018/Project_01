import numpy as np
import pandas as pd
import jieba
import re
from collections import Counter
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
from functools import reduce
import pickle
import networkx
import re


def get_connect_graph_by_text_rank(tokenized_text='', window=3):
    """building word connect graph """
    keywords_graph = networkx.Graph()
    tokeners = tokenized_text.split()
    for ii, t in enumerate(tokeners):
        word_tuples = [(tokeners[connect], t) for connect in range(ii - window, ii + window) if connect >= 0 and connect < len(tokeners)]
        keywords_graph.add_edges_from(word_tuples)
    return keywords_graph


def split_sentence(sentence):
    """split"""
    sentence = ''.join(re.findall(r'[^\s]', sentence))
    pattern = re.compile('[。，,.]')
    split = pattern.sub(' ', sentence).split()
    return split


def get_summarization_simple_with_text_rank(text, constrain=200):
    return get_summarization_simple(text, sentence_ranking_by_text_ranking, constrain)


def sentence_ranking_by_text_ranking(split_sentence):
    """计算sentece的pagerank，并根据值的大小进行排序"""
    sentence_graph = get_connect_graph_by_text_rank(' '.join(split_sentence))
    ranking_sentence = networkx.pagerank(sentence_graph)
    ranking_sentence = sorted(ranking_sentence.items(), key=lambda x: x[1], reverse=True)
    return ranking_sentence


def get_summarization_simple(text, score_fn, consitrain=200):
    # 根据textrank的大小排序，取得前200个字符
    sub_sentence = split_sentence(text)
    ranking_sentence = score_fn(sub_sentence)
    selected_text = set()
    current_text = ''
    for sen, _ in ranking_sentence:
        if len(current_text) < consitrain:
            current_text += sen
            selected_text.add(sen)
        else:
            break
    summarized = []
    for sen in sub_sentence:
        if sen in selected_text:
            summarized.append(sen)
    return summarized


def punctuation_to_sentence(summarization, text):
    # 句子和标点符号的映射，待完善
    result = []
    punctuation = [',', '.', '。', '，']
    decode = [(m.group(), m.span()) for m in re.finditer('|'.join(summarization), text)]
    for sent, span in decode:
        for i in text[span[1]:]:
            if i in punctuation:
                sent += i
                result.append(sent)
                break
    return result

def get_result_simple(text):
    summarization = get_summarization_simple_with_text_rank(text)
    result = punctuation_to_sentence(summarization, text)
    result = (''.join(result)).split('。')
    return '。'.join(result[: -1]) + '。'