import jieba
import re


def cut(sentence):
    pattern = re.compile('[\w+]')
    sentence = ''.join(re.findall(pattern, sentence))
    # return  jieba.lcut(sentence)
    return stop_word(jieba.lcut(sentence))


def stop_word(words):
    with open('/root/project/Project_01//static/stopwords', 'r') as fr:
        stopwords = fr.readlines()

    stopwords = [word.replace('\n', '') for word in stopwords ]

    return [word for word in words if word not in stopwords]

