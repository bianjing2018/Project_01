from gensim.models import Word2Vec, word2vec
import numpy as np
from functools import wraps
from collections import defaultdict
from model.models import NewsChinese
import re, jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import PCA
from pyltp import Parser, Postagger
import os

"""python 使用ltp: https://pyltp.readthedocs.io/zh_CN/latest/api.html"""
CUT_WORD = '../../project_01_data/cut_result'
SAVE_MODEL = './static/save_mode2'
LTP_DATA_DIR = '../../ltp_data'  # ltp模型目录的路径

# 计算向量间的余弦相似度
def compute_cos_similar(vector1,vector2):
    vector1 = np.mat(vector1)
    vector2 = np.mat(vector2)
    num = float(vector1 * vector2)
    denom = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cos = num / denom
    return 0.5 + 0.5 * cos


# 根据维基百科的分词结果进行分词
def train_word2vec_model():
    print('进入——---')
    sentences = word2vec.LineSentence(CUT_WORD)
    print('完成LineSentence')
    model = Word2Vec(sentences=sentences, size=100, min_count=20)
    print('训练model完成')
    model.save(SAVE_MODEL)
    print(model.most_similar(['说']))


# 处理news_chinese 数据库中的数据
def deal_news_chinese():
    i = 0
    datas = NewsChinese.query.all()
    for data in datas:
        try:
            i += 1
            if i%1000 == 0:
                print(i)
            if i >= 78001:
                res = re.findall(r'[\d|\w]+', data.content)
                res = ' '.join(jieba.lcut(''.join(res)))
                with open(CUT_WORD, 'a') as f:
                    f.write(res + '\n')
        except Exception as e:
            continue


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


# 计算输入文本中的句子之间的余弦相似度
def compute_sentence_ftidf(sentences):
    sentences = sentences.split('。') # 对输入语句进行句号分词
    sentences = [jieba.cut(re.findall(r'[\d|\w]+', sentences)) for sentence in sentences]
    c_vector = CountVectorizer()
    c_vector_sentences = c_vector.fit_transform(sentences)
    tf_vector = TfidfTransformer()
    tf_vector_sentences = tf_vector.fit_transform(c_vector_sentences).to_array()  # 生成tfidf
    n_components_size = 14
    for i in range(1, n_components_size):
        pca_tf_vector_sentences = PCA(n_components=i)
        print('pca_tf_vector_sentences valus: {}'.format(i))
        sentence_cos_similar = [compute_cos_similar(pca_tf_vector_sentences[n],
                                                    pca_tf_vector_sentences[n+1])
                                for n in range(pca_tf_vector_sentences.shape[0]-1)]
        print(sentence_cos_similar)
        print('*'*200)


# 使用NER进行 依存分析
def dependency_parse(words):
    # ----------词性标注---------------------------------------------------------------------------
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
    postagger = Postagger()  # 初始化实例
    postagger.load(pos_model_path)  # 加载模型
    # words = ['元芳', '你', '怎么', '看']  # 分词结果
    postags = postagger.postag(words)  # 词性标注
    print('\t'.join(postags))
    postagger.release()  # 释放模型

    # ---------依存分析-----------------------------------------------------------------------------
    par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
    parser = Parser()  # 初始化实例
    parser.load(par_model_path)  # 加载模型
    # words = ['元芳', '你', '怎么', '看']
    # postags = ['nh', 'r', 'r', 'v']
    arcs = parser.parse(words, postags)  # 句法分析
    print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
    parser.release()  # 释放模型




# 判断句子结束为止
def end_of_speech():
    pass


# 根据训练的模型得到 "某人"： "说的言论", 并进行可视化
def visual_image():
    pass

# 可视化前端展示


if __name__ == '__main__':
    words = ['边境', '说', '香港', '是', '中国的']
    dependency_parse(words)
    # deal_news_chinese()
    # train_word2vec_model()
    # model = load_model()
    # model.train(dear_news_chinese())
    # solution, seen = get_say_similar_word('足球', model)
    # print(sorted(solution.items(), key=lambda x:x[1][2], reverse=True))
    # print(seen)

