import numpy as np
from collections import Counter
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
import networkx
from sklearn.decomposition import PCA
import re
import gensim
from views.textrank_word2vec import summarize
from embadding.deal_text import cut
from embadding.base_function import cosine_similar
from sklearn.metrics.pairwise import cosine_similarity
"""https://www.zhongxiaoping.cn/2019/02/25/SIF%E7%AE%97%E6%B3%95%E8%A7%A3%E6%9E%90/#wu-sif-suan-fa-dai-ma-bu-zou sif算法解析"""

WORD_VECTOR = '/root/project/Project_01/static/save_file/save_mode2'

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
        pattern = re.compile('[。？?!！.]')
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
        summarization = self.get_summarization_simple_with_text_rank(text, constrain=len(text) // 10)
        result = self.punctuation_to_sentence(summarization, text)
        result = (''.join(result)).split('。')
        return '。'.join(result[: -1]) + '。'


class SIFSummarization:

    def __init__(self, doc_, title_=None):
        self.model_word_vector = gensim.models.Word2Vec.load(WORD_VECTOR)
        self.doc_ = doc_
        self.title_ = title_
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

    def main(self, flags=1):
        """
        :param flags: 1 使用标题匹配文本相似度；其他值使用sif，每个句子和长文本进行相似度计算
        :return:
        """
        sentence_list = self.doc_.split('。')

        if flags == 1:
            sentence_list.append(self.title_) # 长文本按句号切分句子

        else:
            sentence_list.append(self.doc_) # 将长文本作为句子
        sentence_vector_list = self.sentence_to_vec(sentence_list, embedding_size=100)  # 获得每个句子的句子向量
        special_vector = sentence_vector_list.pop(-1)  # 取出最后一个(标题或长文本)句子向量

        similar_ = []
        for vector in sentence_vector_list:
            similar_.append(cosine_similar(vector, special_vector))

        similar_ = {i: v for i, v in enumerate(similar_)} # 对应cosine value 和 index
        similar_ = sorted(similar_.items(), key=lambda x: x[1], reverse=True)  # 根据cosine value排序
        similar_ = sorted(similar_, key=lambda x: x[1], reverse=True) # 根据

        sorted_score = [i for i, v in similar_[: 3]]  # 取出前3个cosine value 最大的索引

        result = ''
        sorted_score.sort()
        for i in sorted_score:
            result += sentence_list[i]
            result += '。'
        return result, sorted_score, sentence_list


if __name__ == '__main__':
    with open("../static/news.txt", "r", encoding='utf-8') as myfile:
        text = myfile.read().replace('\n', '')
    title_ = '中华人民共和国成立70周年庆祝活动总结会议在京举行 习近平亲切会见庆祝活动筹办工作有关方面代表'

    # 利用textrank
    trs = TextRankSummarization()
    txk_result = trs.get_result_simple(text)
    print(txk_result)


    print('*' * 100)
    print(summarize(text, 2))

    sif = SIFSummarization(text)
    sif_result_, sorted_score_, sentence_list_ = sif.main(flags=0)
    print('*' * 100)
    print(sif_result_, end='\n')
    #
    sif = SIFSummarization(text, title_)
    sif_result, sorted_score, sentence_list = sif.main(flags=1)
    print(sif_result, sorted_score, sep='\n')








