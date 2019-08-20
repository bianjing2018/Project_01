from gensim.models import Word2Vec, word2vec
import numpy as np
from functools import wraps
from collections import defaultdict
from model.models import NewsChinese
import re, jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from pyltp import Parser, Postagger
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
import wordcloud
import matplotlib.pyplot as plt
import scipy
from matplotlib.pyplot import imread
from sklearn.metrics.pairwise import cosine_similarity



"""python 使用ltp: https://pyltp.readthedocs.io/zh_CN/latest/api.html"""
CUT_WORD = '../../project_01_data/cut_result' # 所有词的分词结果包含维基百科和新闻数据
NEWCUTWORD = '../../project_01_data/news_cut_word'  # 新闻分词结果，带标签
SAVE_MODEL = './static/save_file/save_mode2'
LTP_DATA_DIR = '../../ltp_data'  # ltp模型目录的路径
SAVE_MODEL_NB = './static/save_file/save_mode_nb'


# 计算向量间的余弦相似度
def compute_cos_similar(vector1,vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    num = np.sum(vector1  * vector2)
    denom = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cos = num / denom
    return 0.5 + 0.5 * cos


# 根据维基百科的分词结果进行分词
def train_word2vec_model(path, save_path):
    print('进入——---')
    sentences = word2vec.LineSentence(path)
    print('完成LineSentence')
    model = Word2Vec(sentences=sentences, size=100, min_count=20)
    print('训练model完成')
    model.save(save_path)
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
            res = re.findall(r'[\d|\w]+', data.content)
            res_class = re.findall(r'"type":"(.*)","site".*', data.feature)[0]
            if res and res_class:
                res = ' '.join(i for i in jieba.lcut(''.join(res)) if i != 'n')
                res += ' {}'.format(res_class)
            with open(NEWCUTWORD, 'a') as f:
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
                if '{}-{}'.format(current, sw) not in solution and len(solution) < 200:
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
    result = []
    for sentence in sentences:
        sentence = ' '.join(re.findall(r'[\d|\w]+', sentence))
        sentence = ' '.join(jieba.lcut(sentence))
        result.append(sentence)
    sentences = result
    c_vector = CountVectorizer()
    c_vector_sentences = c_vector.fit_transform(sentences)
    tf_vector = TfidfTransformer()
    tf_vector_sentences = tf_vector.fit_transform(c_vector_sentences).toarray()  # 生成tfidf

    sentence_cos_similar = [compute_cos_similar(tf_vector_sentences[n],
                                                tf_vector_sentences[n + 1])
                            for n in range(tf_vector_sentences.shape[0] - 1)]
    return sentence_cos_similar, sentences

    # for i in range(1, 4):
    #     pca = PCA(n_components=i)
    #     pca_tf_vector_sentences = pca.fit_transform(tf_vector_sentences)
    #     print('pca_tf_vector_sentences valus: {}'.format(i))
    #     sentence_cos_similar = [compute_cos_similar(pca_tf_vector_sentences[n],
    #                                                 pca_tf_vector_sentences[n+1])
    #                             for n in range(pca_tf_vector_sentences.shape[0]-1)]
    #     print(sentence_cos_similar)
    #     print('*'*200)


def wordcloud_(words):
    # image_ground = scipy.mis('/Users/bj/Desktop/chinamap.jpg')
    image_ground = imread('./static/img/chinamap.jpg')
    wd = wordcloud.WordCloud(
        font_path='/usr/local/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf',
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


# 可视化
class DataGraphDisplay:
    def __init__(self, key=None):
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
    def __init__(self, path='../../ltp_data', words=[]):
        self.LTP_DATA_DIR = path  # ltp模型目录的路径
        self.arcs = None
        self.words = words

    def deal_sentence(self, sentence):
        # 处理句子，并且分词
        pass

    def get_word_xing(self, words):
        # 词性标注
        pos_model_path = os.path.join(self.LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
        postagger = Postagger()  # 初始化实例
        postagger.load(pos_model_path)  # 加载模型
        # words = ['元芳', '你', '怎么', '看']  # 分词结果
        postags = postagger.postag(words)  # 词性标注
        print('\t'.join(postags))
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
        print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
        parser.release()  # 释放模型
        self.arcs = arcs

    def get_HED(self):
        # get HED
        root = None
        for i, arc in enumerate(self.arcs):
            if arc.relation == 'HED' and arc.head == 0:
                root = (i, arc.relation, self.words[i])
        return root

    def get_word(self, head, wtype):
        # get related word
        for i, arc in enumerate(self.arcs):
            if (arc.head - 1) == head and arc.relation == wtype:
                return self.words[i], i
        return 'nan', 'nan'

    def get_main(self):
        self.get_word_depend(self.words)
        root = self.get_HED()
        if root:
            hed = root[2] # 谓语
            sbv, sbv_i = self.get_word(root[0], 'SBV')  # 获取主语 root[0] 为谓语动词的索引值
            # vob, vob_i = self.get_word(root[0], 'VOB')  # 获取宾语
            # fob, fob_i = self.get_word(root[0], 'FOB')  # 后置宾语
            # adv, adv_i = self.get_word(root[0], 'ADV')  # 副词做状语
            # pob, pod_i = self.get_word(adv_i, 'POB')

            zhuyu = [sbv]
            weiyu = [hed]
            self.words.remove(sbv)
            self.words.remove(hed)
            print('{}  {}  {}'.format(' '.join(zhuyu), ' '.join(weiyu), ' '.join(self.words)))

# str = "新华社兰州6月3日电（王衡、徐丹）记者从甘肃省交通运输厅获悉，甘肃近日集中开建高速公路、普通国省道、服务区、物流园等涉及12个市州的35个重点交通建设项目，总投资达693.8亿元。\n其中，投资最大的是5条高速和一级公路项目，投资604亿元，包括兰州高速二环公路（清水驿至苦水段、忠和至河口段）、G1816乌海－玛沁高速景泰至中川机场段、G8513平凉－绵阳高速平凉（华亭）至天水段、S216线平凉至华亭一级公路和G341线白银至中川机场段。\n甘肃省交通运输厅介绍，此次集中开建35个重点交通建设项目，在进一步完善路网结构的同时，有助于促进甘肃经济运行趋稳向好、确保完成全年固定资产投资目标任务。（完）"
# words = jieba.lcut(str)
# PD = ParseDepend(words=words)
# PD.get_main()


# 文本分类模型
class TrainNewsClassModel:

    def __init__(self, data=[], target=[]):
        self.data = data # 新闻数据
        self.target = target # 类别标签
        self.hash_encode = defaultdict(int) # 编码类别
        self.hash_decode = defaultdict(str) # 解码类别

    def target_mapping(self, news_category):
        """标签映射"""
        if news_category not in self.hash_encode:
            if not self.hash_encode:
                self.hash_encode[news_category] = 0
                self.hash_decode[0] = news_category
            else:
                value = max(self.hash_encode.values()) + 1
                self.hash_encode[news_category] = value
                self.hash_decode[value] = news_category
        else:
            pass

    def get_news_chinese(self):
        """从数据库中获取新闻数据和标签"""
        DATAS = NewsChinese.query.all()
        for data in DATAS:
            try:
                news = re.findall(r'[\d|\w]+', data.content)
                news_category = re.findall(r'"type":"(.*)","site".*', data.feature)[0]
                if news and news_category:
                    res = ' '.join(i for i in jieba.lcut(''.join(news)) if i != 'n')
                    self.data.append(res)
                    self.target_mapping(news_category)
                    self.target.append(self.hash_encode[news_category])
            except Exception as e:
                continue
        with open('pickle_hash_encode', 'wb') as f:
            pickle.dump(self.hash_encode, f)

        with open('pickle_hash_decode', 'wb') as f:
            pickle.dump(self.hash_decode, f)

    def get_tfidf_vector(self, x_train):
        """词向量tfidf 并保存"""
        cv_m = CountVectorizer()
        cv_x_train = cv_m.fit_transform(x_train)
        tf_m = TfidfTransformer()
        tf_x_train = tf_m.fit_transform(cv_x_train)
        with open('pickle_cv_m', 'wb') as f:
            pickle.dump(cv_m, f)

        with open('pickle_cv_x_train', 'wb') as f:
            pickle.dump(cv_m, f)
        return tf_x_train

    def train_model(self):
        """训练模型并打印评分"""
        print(self.data)
        print(self.target)
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.target, train_size=0.8, random_state=7)
        tf_x_train = self.get_tfidf_vector(x_train)
        model = MultinomialNB()
        precision_weighted = cross_val_score(model, tf_x_train, y_train, cv=5, scoring='precision_weighted').mean()
        recall_weighted = cross_val_score(model, tf_x_train, y_train, cv=5, scoring='recall_weighted').mean()
        f1_weighted = cross_val_score(model, tf_x_train, y_train, cv=5, scoring='f1_weighted').mean()
        model.fit(tf_x_train, y_train)
        with open('pickle_cv_m', 'rb') as f:
            cv_m = pickle.load(f)

        with open('pickle_cv_x_train', 'rb') as f:
            cv_x_train = pickle.load(f)

        cv_x_test = cv_m.transform(x_test)
        tf_m = TfidfTransformer()
        tf_x_test = tf_m.fit_transform(cv_x_test)
        y_hat = model.predict(tf_x_test)
        result = y_hat == y_test
        result = [r for r in result if r]
        score = {'查准率': precision_weighted,
                 '召回率': recall_weighted,
                 'F1得分': f1_weighted,
                 '正确率': len(result)/ y_hat.size}
        print(score)
        with open('pickle_model', 'wb') as f:
            pickle.dump(model, f)

    def model_predict(self, article):
        """预测model"""
        with open('pickle_cv_m', 'rb') as f:
            cv_m = pickle.load(f)

        with open('pickle_cv_x_train', 'rb') as f:
            cv_x_train = pickle.load(f)

        with open('pickle_model', 'rb') as f:
            model = pickle.load(f)

        with open('pickle_hash_encode', 'rb') as f:
            hash_encode = pickle.load(f)

        with open('pickle_hash_decode', 'rb') as f:
            hash_decode = pickle.load(f)

        article = re.findall(r'[\d|\w]+', article)
        article = [' '.join(jieba.lcut(''.join(article)))]
        cv_article = cv_m.transform(article)
        tf_m = TfidfTransformer()
        tf_article = tf_m.fit_transform(cv_article)
        hash_decode_key = model.predict(tf_article)[0]
        print(hash_decode[hash_decode_key])



if __name__ == '__main__':
    pass
    # obj = TrainNewsClassModel()
    # obj.get_news_chinese()
    # obj.train_model()
    # obj.model_predict('新华社照片，外代，2017年6月7日\n（外代二线）足球——国际友谊赛：德国平丹麦\n6月6日，丹麦队门将伦诺门前救险。\n当日，在丹麦布隆德比进行的一场国际足球友谊赛中，德国队1比1战平丹麦队。\n新华社/欧新\n')
    # nb_news_class()
    # words = ['他', '什么', '书', '都', '读']
    #
    # dependency_parse(words)
    # deal_news_chinese()
    # train_word2vec_model()
    # model = load_model()
    # # print(model.similarity('说', '活捉'))
    # # m = odel.train(dear_news_chinese())
    # solution, seen, quanzhong= get_say_similar_word('说', model)
    # words = {}
    # i = 0
    # print(solution)
    # word_cloud = {}
    # for k, v in solution.items():
    #     if v[0] in quanzhong:
    #         word_cloud[v[0]] = quanzhong[v[0]]
    # for k, v in sorted(dict(quanzhong).items(), key=lambda x: x[1], reverse=True):
    #     if k in seen:
    #         words[k] = v
    #         i += 1


    # wordcloud_(word_cloud)
    # print(seen)
    # print(quanzhong)

    # def cut_word(sentences):
    #     sentences = sentences.split('。')  # 对输入语句进行句号分词
    #     result = []
    #     for sentence in sentences:
    #         sentence = ''.join(re.findall(r'[\d|\w]+', sentence))
    #         sentence = jieba.lcut(sentence)
    #         result.extend(sentence)
    #     sentences = result
    #     return sentences
    # words = cut_word("新华社照片，外代，2017年6月7日\n（外代二线）网球——法网：奥斯塔片科晋级半决赛\n6月6日，拉脱维亚选手奥斯塔片科（左）赛后与丹麦选手沃兹尼亚奇握手。"
    #                        "\n当日，在巴黎举行的法国网球公开赛女单四分之一决赛中，拉脱维亚选手奥斯塔片科2比1战胜丹麦选手沃兹尼亚奇，晋级半决赛。"
    #                        "\n新华社/欧新\n新")
    # dependency_parse(words)
    #
    sentence_cos_similar, sentences = compute_sentence_ftidf("新华社照片，外代，2017年6月7日\n（外代二线）网球——法网：奥斯塔片科晋级半决赛\n6月6日，拉脱维亚选手奥斯塔片科（左）赛后与丹麦选手沃兹尼亚奇握手。"
                           "\n当日，在巴黎举行的法国网球公开赛女单四分之一决赛中，拉脱维亚选手奥斯塔片科2比1战胜丹麦选手沃兹尼亚奇，晋级半决赛。"
                           "\n新华社/欧新\n新")
    print(sentence_cos_similar)
    print(sentences)
