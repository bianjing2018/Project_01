import numpy as np
from collections import Counter
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
import networkx
from sklearn.decomposition import PCA
import re
from embadding.deal_text import cut
from embadding.base_function import cosine_similar
from sklearn.metrics.pairwise import cosine_similarity
"""https://www.zhongxiaoping.cn/2019/02/25/SIF%E7%AE%97%E6%B3%95%E8%A7%A3%E6%9E%90/#wu-sif-suan-fa-dai-ma-bu-zou sif算法解析"""
WORD_VECTOR = '../static/save_file/fasttext_size100.model'


class TextRankSummarization:
    """
    利用textRank实现的文本摘要
    """
    def __init__(self):
        pass

    def get_connect_graph_by_text_rank(self, tokenized_text='', window=3):
        """building word connect graph """
        keywords_graph = networkx.Graph()
        tokeners = tokenized_text.split()
        for ii, t in enumerate(tokeners):
            word_tuples = [(tokeners[connect], t) for connect in range(ii - window, ii + window) if connect >= 0 and connect < len(tokeners)]
            keywords_graph.add_edges_from(word_tuples)
        return keywords_graph

    def split_sentence(self, sentence):
        """split"""
        sentence = ''.join(re.findall(r'[^\s]', sentence))
        pattern = re.compile('[。，,.]')
        split = pattern.sub(' ', sentence).split()
        return split

    def get_summarization_simple_with_text_rank(self, text, constrain=200):
        return self.get_summarization_simple(text, self.sentence_ranking_by_text_ranking, constrain)

    def sentence_ranking_by_text_ranking(self, split_sentence):
        """计算sentece的pagerank，并根据值的大小进行排序"""
        sentence_graph = self.get_connect_graph_by_text_rank(' '.join(split_sentence))
        ranking_sentence = networkx.pagerank(sentence_graph)
        ranking_sentence = sorted(ranking_sentence.items(), key=lambda x: x[1], reverse=True)
        return ranking_sentence

    def get_summarization_simple(self, text, score_fn, consitrain=200):
        # 根据textrank的大小排序，取得前200个字符
        sub_sentence = self.split_sentence(text)
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

    def punctuation_to_sentence(self, summarization, text):
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

    def get_result_simple(self, text):
        summarization = self.get_summarization_simple_with_text_rank(text)
        result = self.punctuation_to_sentence(summarization, text)
        result = (''.join(result)).split('。')
        return '。'.join(result[: -1]) + '。'


class SIFSummarization:

    def __init__(self, doc_):
        self.model_word_vector = FastText.load(WORD_VECTOR)
        self.doc_ = doc_
        self.words = cut(doc_) # 对整篇文章进行分词
        self.counter = Counter(self.words)   # 对分词结果进行Counter，方便计算词频

    def get_word_frequency(self, word):
        return self.counter[word] / len(self.words)

    def sentence_to_vec(self, sentence_list, embedding_size=100, a: float = 1e-3):
        sentence_set = []
        for sentence in sentence_list:
            vs = np.zeros(embedding_size)  # add all word2vec values into one vector for the sentence
            word_list = cut(sentence)
            sentence_length = len(word_list)
            # 这个就是初步的句子向量的计算方法
            try:
                if word_list and sentence_length:
                    for word in word_list:
                        if word in self.model_word_vector:
                            a_value = a / (a + self.get_word_frequency(word))  # smooth inverse frequency, SIF
                            vs = np.add(vs, np.multiply(a_value, self.model_word_vector[word]))  # vs += sif * word_vector
                        else:
                            continue

                    vs = np.divide(vs, sentence_length)  # weighted average
                    sentence_set.append(vs)  # add to our existing re-calculated set of sentences
                else:
                    continue
            except:
                continue
        # calculate PCA of this sentence set,计算主成分
        pca = PCA()
        # 使用PCA方法进行训练
        pca.fit(np.array(sentence_set))
        # 返回具有最大方差的的成分的第一个,也就是最大主成分,
        # components_也就是特征个数/主成分个数,最大的一个特征值
        u = pca.components_[0]  # the PCA vector
        # 构建投射矩阵
        u = np.multiply(u, np.transpose(u))  # u x uT
        # judge the vector need padding by wheather the number of sentences less than embeddings_size
        # 判断是否需要填充矩阵,按列填充
        if len(u) < embedding_size:
            for i in range(embedding_size - len(u)):
                # 列相加
                u = np.append(u, 0)  # add needed extension for multiplication below

        # resulting sentence vectors, vs = vs -u x uT x vs
        sentence_vecs = []
        for vs in sentence_set:
            sub = np.multiply(u, vs)
            sentence_vecs.append(np.subtract(vs, sub))
        return sentence_vecs

    def compute_similar_by_cosine(self, sentence_vector_list):
        doc_sentence = sentence_vector_list.pop(-1)
        square_doc = np.sqrt(np.sum(np.square(doc_sentence)))
        similar = []
        for i, sentence_vector in enumerate(sentence_vector_list):
            up = np.dot(sentence_vector, doc_sentence)
            down = np.sqrt(np.sum(np.square(sentence_vector))) + square_doc
            similar.append(up / down)
        similar_ = {i: v for i, v in enumerate(similar)}
        return similar_


if __name__ == '__main__':
    text = "中国首富的位子几乎每年都在变动，这三年来已经换了三位首富。日前，2019年的中国富豪榜正式发布，其中马云又一次成了中国最有钱的。马化腾身价2600亿元排名第二位，许家印以2100亿元的资产，" \
           "排名第三位。作为“新科”首富，马云不仅让自己成为千亿级别的富豪，更是让12位下属员工也都成了身价亿万的富豪。马云以2750亿元的财富再次登顶富豪榜榜首，成为中国新首富。与之前的榜单相比，" \
           "马云家族的财富仅仅增加了50亿元。虽然财富增幅不大，但是这在很大程度上意味着马云所建立的阿里巴巴帝国在市场中的地位日渐稳固。电商等核心业务的稳健发展与市场扩张，是阿里巴巴保持较高市" \
           "值的关键，也是马云家族财富稳固的重要保障。除了个人身价增长外，阿里系的12名员工也都登上了富豪榜，这些人都是马云的下属，手中也都握有阿里的股票。可以预见的是，虽然创始人已经正式退休，" \
           "但是阿里的未来的发展方向不会变，甚至可能会比马云主政时期发展的更为迅猛。现在的阿里可以说是人才济济，良将如云，根本不用为人才断层所发愁。近日流传，马云将收购刷爆朋友圈的安贸通APP，" \
           "一双双上千元的耐克阿迪运动鞋，上万元Lv香奈儿等奢侈品，在安贸通APP上不到100元，质量却堪比正品？而微信一对一交流女大学生店主，或是马云又开始尝试社交电商了吗？中国首富的位置其实说是财富，" \
           "在一定的程度上面也影响了很多人，马云的口碑是十分不错的，不仅在向全世界宣传中国，时时刻刻不忘了造福百姓，宣传国家的形象，影响改变着我们的生活，让我们的生活走向更为发展的路程，" \
           "马云成为中古首富想必也是大家希望的。"
    sentence_list = text.split('。')
    title_ = '中国新首富诞生：身家2750亿力压马化腾，许家印排名仅第3'
    sentence_list.append(text)
    sentence_list.append(title_)
    ss = SIFSummarization(text)
    sentence_vector_list = ss.sentence_to_vec(sentence_list)
    titleVector = sentence_vector_list.pop(-1)
    docVector = sentence_vector_list.pop(-1)

    # 利用sif
    similar = []
    for vector in sentence_vector_list:
        similar.append(cosine_similar(vector, docVector))
    similar = {i: v for i, v in enumerate(similar)}
    # similar = ss.compute_similar_by_cosine(sentence_vector_list)
    # print(similar)
    sss = sorted(similar.items(), key=lambda x: x[1], reverse=True)
    sss = sorted(sss, key=lambda x: x[1], reverse=True)
    print(sss)
    summ = ''
    for i, v in sss:
        summ += sentence_list[i]
        summ += ' '
        if len(summ) >= 100:
            break
    print(summ)


    # 利用 标题
    similar = []
    for vector in sentence_vector_list:
        similar.append(cosine_similar(vector, titleVector))
    similar = {i: v for i, v in enumerate(similar)}
    # similar = ss.compute_similar_by_cosine(sentence_vector_list)
    # print(similar)
    sss = sorted(similar.items(), key=lambda x: x[1], reverse=True)
    sss = sorted(sss, key=lambda x: x[1], reverse=True)
    print(sss)
    summ = ''
    for i, v in sss:
        summ += sentence_list[i]
        summ += ' '
        if len(summ) >= 100:
            break
    print(summ)

    # 利用textrank
    trs = TextRankSummarization()
    result = trs.get_result_simple(text)
    print(result)






