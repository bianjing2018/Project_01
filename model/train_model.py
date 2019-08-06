from gensim.models import Word2Vec, word2vec
import numpy as np

CUT_WORD = '../static/cut_result'
SAVE_MODEL = '../static/save_model'


# 加载word2vec
def train_word2vec_model():
    sentences = word2vec.LineSentence(CUT_WORD)
    model = Word2Vec(sentences=sentences, size=100, min_count=20)
    model.save(SAVE_MODEL)
    model.most_similar(['说'])


# 找到'说的'近义词：不能根据自带函数 most_similar查询，需要利用预选相似度，通过 bfs搜索查询，每个词附上权值。
def get_say_similar_word():
    pass


# 使用NER进行 依存分析
def dependency_parse():
    pass


# 判断句子结束为止
def end_of_speech():
    pass


# 根据训练的模型得到 "某人"： "说的言论", 并进行可视化
def visual_image():
    pass

# 可视化前端展示


