from gensim.models import Word2Vec, word2vec
import numpy as np
from functools import wraps
from collections import defaultdict
from model.models import NewsChinese
import re, jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from pyltp import Parser, Postagger, NamedEntityRecognizer
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
import wordcloud
import matplotlib.pyplot as plt
import scipy
from matplotlib.pyplot import imread
from sklearn.metrics.pairwise import cosine_similarity


"""python 使用ltp: https://pyltp.readthedocs.io/zh_CN/latest/api.html"""
CUT_WORD = '../project_01_data/cut_result' # 所有词的分词结果包含维基百科和新闻数据
NEWCUTWORD = '../project_01_data/news_cut_word'  # 新闻分词结果，带标签
SAVE_MODEL = './static/save_file/save_mode2'
LTP_DATA_DIR = '../ltp_data'  # ltp模型目录的路径
SAVE_MODEL_NB = './static/save_file/save_mode_nb'


# 计算向量间的余弦相似度
def compute_cos_similar(vector1,vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    up = np.sum(vector1 * vector2)
    vector1_mo = np.sqrt(np.sum(np.square(vector1)))
    vector2_mo = np.sqrt(np.sum((np.square(vector2))))
    down = vector1_mo * vector2_mo
    if not down:
        return 0.000000001
    return up / down


# 计算输入文本中的句子之间的余弦相似度
def compute_sentence_ftidf(sentences):
    # sentences = sentences.split('。') # 对输入语句进行句号分词
    result = []
    for sentence in sentences:
        sentence = ' '.join(re.findall(r'[\d|\w]+', sentence))
        sentence = ' '.join([s for s in jieba.lcut(sentence) if s != 'n'])
        result.append(sentence)
    sentences = result
    c_vector = CountVectorizer()
    c_vector_sentences = c_vector.fit_transform(sentences)
    tf_vector = TfidfTransformer()
    tf_vector_sentences = tf_vector.fit_transform(c_vector_sentences).toarray()  # 生成tfidf

    sentence_cos_similar = [compute_cos_similar(tf_vector_sentences[n],
                                                tf_vector_sentences[n + 1])
                            for n in range(tf_vector_sentences.shape[0] - 1)]
    max_index = {'max_index': i + 1 for i, s in enumerate(sentence_cos_similar) if s > 0.5}
    max_index = max_index['max_index'] + 1
    return sentence_cos_similar, sentences[: max_index]


# 可视化
class DataGraphDisplay:
    def __init__(self, key='说'):
        self.model = Word2Vec.load(SAVE_MODEL)
        self.key = key

    def get_say_similar_word(self):
        model = self.model
        key = self.key
        say_similar_words = [(key, 0)]
        seen = []
        solution = {'{}-""'.format(key): (key, 0, 1)}
        max_size = 500
        db_table = {}  # 路径经历过的词表
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
                    say_similar_words += word_weights_similar  # 宽度优先
                    if '{}-{}'.format(current, sw) not in solution and len(solution) < 200:
                        solution['{}-{}'.format(current, sw)] = (
                        sw, compute_cos_similar(model[current], model[sw]))  # 记录每一个词的 权重及相似度
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

    def word_cloud(self):
        solution, seen, quanzhong = self.get_say_similar_word()
        words = {}
        for k, v in solution.items():
            if v[0] in quanzhong:
                words[v[0]] = quanzhong[v[0]]
        image_ground = imread('./static/img/chinamap.jpg')
        wd = wordcloud.WordCloud(
            font_path='./static/DejaVuSans.ttf',
            background_color='white',
            mask=image_ground,
            max_words=2000,
            max_font_size=100,
            random_state=2)
        image_colors = wordcloud.ImageColorGenerator(image_ground)
        wd.fit_words(words)
        wd.to_file('./static/img/wordcloud.jpg')
        plt.imshow(wd)
        plt.axis("off")
        plt.show()
        plt.imsave()


# 依存分析
class ParseDepend:
    """
    sentences: ["a b c", "d, e, f"]
    """
    def __init__(self, path='../ltp_data', sentences=[]):
        self.LTP_DATA_DIR = path  # ltp模型目录的路径
        self.arcs = None
        self.sentences = sentences

    def deal_sentence(self, sentences):
        # 处理句子，并且分词
        sentences = sentences.split('。')
        sentences = [''.join(re.findall(r'[\d|\w]+', sen)) for sen in sentences]
        sentences = [' '.join([l for l in jieba.lcut(sentence) if l != 'n']) for sentence in sentences]
        self.sentences= sentences

    def my_cut(self, sentences):
        words = []
        for s in sentences:
            words.extend(jieba.lcut(s))
        return words

    def get_word_xing(self, words):
        # 词性标注
        pos_model_path = os.path.join(self.LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
        postagger = Postagger()  # 初始化实例
        postagger.load(pos_model_path)  # 加载模型
        # words = ['元芳', '你', '怎么', '看']  # 分词结果
        postags = postagger.postag(words)  # 词性标注
        postagger.release()  # 释放模型
        return postags

    def get_word_depend(self, words):
        # 依存分析
        postags = self.get_word_xing(words)
        par_model_path = os.path.join(self.LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
        parser = Parser()  # 初始化实例
        parser.load(par_model_path)  # 加载模型
        # words = ['元芳', '你', '怎么', '看']
        # postags = ['nh', 'r', 'r', 'v']
        arcs = parser.parse(words, postags)  # 句法分析
        # print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
        parser.release()  # 释放模型
        self.arcs = arcs

    def get_HED(self, words):
        # get HED
        root = None
        for i, arc in enumerate(self.arcs):
            if arc.relation == 'HED' and arc.head == 0:
                root = (i, arc.relation, words[i])
        return root

    def get_ner(self, words):
        # 命名实体识别
        postags = self.get_word_xing(words)
        ner_model_path = os.path.join(self.LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
        recognizer = NamedEntityRecognizer()  # 初始化实例
        recognizer.load(ner_model_path)  # 加载模型

        netags = recognizer.recognize(words, postags)  # 命名实体识别
        recognizer.release()  # 释放模型
        return '\t'.join(netags)

    def get_word(self, head, wtype, sentence):
        # get related word
        for i, arc in enumerate(self.arcs):
            if (arc.head - 1) == head and arc.relation == wtype:
                return sentence[i], i
        return 'nan', 'nan'

    def get_main(self):
        result = []
        for i, sentence in enumerate(self.sentences):
            sentence = sentence.split(' ')
            self.get_word_depend(sentence)   # 依存分析
            root = self.get_HED(sentence)   # 获取 谓语root
            netags = self.get_ner(sentence)  # 命名实体获取
            if root:
                hed = root[2]    # 谓语
                sbv, sbv_i = self.get_word(root[0], 'SBV', sentence)  # 获取主语 root[0] 为谓语动词的索引值
                zhuyu = [sbv]
                weiyu = [hed]
                hed_index = sentence.index(hed)
                words_r = sentence[hed_index+1:]
                result.append([' '.join(zhuyu), ' '.join(weiyu), ' '.join(words_r)])
        return result

