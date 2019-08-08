from gensim.models import Word2Vec, word2vec
import numpy as np
from functools import wraps
from collections import defaultdict

"""python 使用ltp: https://pyltp.readthedocs.io/zh_CN/latest/api.html"""
CUT_WORD = '../../project_01_data/cut_result'
SAVE_MODEL = '../static/save_mode2'


# 计算向量间的余弦相似度
def compute_cos_similar(vector1,vector2):
    up = np.sum(vector1 * vector1) + 1
    down = np.sqrt(np.sum(np.square(vector1))) * np.sqrt(np.sum(np.square(vector2))) + len(vector1)
    return up/down


def train_word2vec_model():
    print('进入——---')
    sentences = word2vec.LineSentence(CUT_WORD)
    print('完成LineSentence')
    model = Word2Vec(sentences=sentences, size=100, min_count=20)
    print('训练model完成')
    model.save(SAVE_MODEL)
    print(model.most_similar(['说']))


# 加载模型
def load_model():
    model = Word2Vec.load(SAVE_MODEL)
    return model


# 找到'说的'近义词：不能根据自带函数 most_similar查询，需要利用余弦相似度，通过 bfs搜索查询，每个词附上权值。
def get_say_similar_word(key='说', model=None):
    say_similar_words = [(key, 0)]
    seen = []
    solution = {'{}-""'.format(key):(key, 0, 1)}
    max_size = 500
    db_table = {} # 路径经历过的词表
    quanzhong = defaultdict(int)
    while say_similar_words and len(seen) < 500:
        say = say_similar_words.pop(0)
        current = say[0]
        similar_words = model.most_similar([current], topn=20)
        add_dp_table = []  # 记录每一个 current下关联的相似词
        if current not in db_table:
            for sw, similar in similar_words:
                if sw in seen: continue
                word_weights_similar = [(sw, compute_cos_similar(model[current], model[sw]))]
                quanzhong[sw] += 1
                add_dp_table += word_weights_similar
                say_similar_words += word_weights_similar # 宽度优先
                if '{}-{}'.format(current, sw) not in solution:
                    solution['{}-{}'.format(current, sw)] = (sw, compute_cos_similar(model[current], model[sw])) # 记录每一个词的 权重及相似度
            db_table[current] = add_dp_table
            seen.append(current)
            say_similar_words.sort(key=lambda x: x[1], reverse=True)
        else:
            for sw, i in db_table[current]:
                quanzhong[sw] += 1
            say_similar_words += db_table[current]
            seen.append(current)
            say_similar_words.sort(key=lambda x: x[1], reverse=True)
    return solution, seen, quanzhong


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


if __name__ == '__main__':
    train_word2vec_model()
    # model = load_model()
    # solution, seen = get_say_similar_word('足球', model)
    # print(sorted(solution.items(), key=lambda x:x[1][2], reverse=True))
    # print(seen)

