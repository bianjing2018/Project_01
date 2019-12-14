from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
from views.textrank_word2vec import summarize
from ai_baidu_api.aip import nlp
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from collections import Counter
import networkx
from sklearn.decomposition import PCA
from gensim import corpora, models, similarities
import re
import gensim
import os
import jieba
from sklearn.feature_extraction.text import CountVectorizer
import operator
from textrank4zh import TextRank4Sentence
from tools.deal_text import cut
from tools.metrics import rouge_n
from tools.base_function import cosine_similar, sentence_to_vec
import warnings
"""https://www.zhongxiaoping.cn/2019/02/25/SIF%E7%AE%97%E6%B3%95%E8%A7%A3%E6%9E%90/#wu-sif-suan-fa-dai-ma-bu-zou sif算法解析"""

warnings.filterwarnings('ignore')

if os.path.exists('/root/flag_server'):
    WORD_VECTOR = '/root/project/Project_01/static/save_file/save_mode2'
elif os.path.exists('/Volumes/Samsung_T5/'):
    WORD_VECTOR = "/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_word2Vec/THUCNews_word2Vec_128.model"
elif os.path.exists('/Users/haha'):
    WORD_VECTOR = '/Users/haha/Desktop/Project_01/static/save_file/save_mode2'
elif os.path.exists('/Users/bj') :
    WORD_VECTOR = '/Users/bj/Desktop/Documents/Project_01/static/save_file/save_mode2'

W2V_MODEL = gensim.models.Word2Vec.load(WORD_VECTOR)

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

        sentence = ''.join(re.findall(r'[^\s]', self.doc_))
        pattern = re.compile('[。？?!！.]')
        sentence_list = pattern.sub(' ', sentence).split()

        # sentence_list = self.doc_.split('。')

        if flags == 1:
            sentence_list.append(self.title_) # 长文本按句号切分句子

        else:
            sentence_list.append(self.doc_) # 将长文本作为句子
        sentence_vector_list = self.sentence_to_vec(sentence_list, embedding_size=128)  # 获得每个句子的句子向量
        special_vector = sentence_vector_list.pop(-1)  # 取出最后一个(标题或长文本)句子向量

        similar_ = []
        for vector in sentence_vector_list:
            similar_.append(cosine_similar(vector, special_vector))

        similar_ = {i: v for i, v in enumerate(similar_)} # 对应cosine value 和 index
        similar_ = sorted(similar_.items(), key=lambda x: x[1], reverse=True)  # 根据cosine value排序
        similar_ = sorted(similar_, key=lambda x: x[1], reverse=True) # 根据

        sorted_score = [i for i, v in similar_[: 3]]  # 取出前3个cosine value 最大的索引
        # return sentence_list[sorted_score[0]]
        result = ''
        sorted_score.sort()
        for i in sorted_score:
            # return sentence_list[i]
            result += sentence_list[i]
            result += '。'
        return result, sorted_score, sentence_list


class MMRSummarization:
    def __init__(self):
        self.word2vec = None
        self.text = None

    def sentence_to_vector(self, sentence_list):
        return sentence_to_vec(corpus=self.text, sentence_list=sentence_list, W2V_MODEL=self.word2vec)

    def compute_qdscore(self):
        sentence_list = self.split_sentence(self.text)  # [sen1, sen2, ...]
        sentence_list.append(''.join(sentence_list))
        QDsocre, index_solution = {}, {}
        sentence_list_vector = self.sentence_to_vector(sentence_list)
        special_vector = sentence_list_vector.pop(-1) # 长文本
        for inx, vector in enumerate(sentence_list_vector):
            qdscore = cosine_similar(vector, special_vector)
            QDsocre[sentence_list[inx]] = qdscore
            index_solution[sentence_list[inx]] = inx
        return QDsocre, index_solution, sentence_list_vector, sentence_list

    def stop_word(self, path, sentence):
        with(path, 'r') as fr:
            stopwords = [line.replace('\n', '') for line in fr.read() if line != '\n']
        res = [[s for s in sent if s not in stopwords] for sent in sentence ]
        return res

    def split_sentence(self, sentence):
        sentence = ''.join(re.findall(r'[^\s]', sentence))
        pattern = re.compile('[。？?!！.]')
        split = pattern.sub(' ', sentence).split()
        return split

    def MMR(self, sentence=None, alpha=0.7, max_len=200):
        """
        main function: alpha * similar(Q, D) - (1-alpha) * max(similar(D_i, D_j))
        first step: compute Q, D similar
        second step: 选择第一步中得分最高的作为结果结果集中的一个子集
        third step: 遍历第一步的句子得分结果，得到每个句子的MMR得分。并取得最大MMR得分加入摘要集合中
        sentence: 文本内容
        stopwords: 停止词path
        embedding: 是否使用word2vec，默认onehot
        """
        self.text = sentence
        self.word2vec = W2V_MODEL
        QDsocre, index_solution, sentence_list_vector, sentence_list = self.compute_qdscore() #
        summarize_set = list()
        while 'summarize_set_str' not in locals() or len(summarize_set_str) < max_len:
            if not summarize_set:
                summarize_set.append(sentence_list[0])
                # max_score = sorted(QDsocre.items(), key=operator.itemgetter(1), reverse=True)[0][0]
                # summarize_set.append(max_score)
            MMRscore = {}
            for sen in QDsocre.keys():
                if sen not in summarize_set:
                    summarize_vectors = [sentence_list_vector[index_solution[summary_str]]
                                             for summary_str in summarize_set]
                    sen_vector = sentence_list_vector[index_solution[sen]]
                    mmrscore = alpha * QDsocre[sen] - ((1 - alpha) * max(
                        [cosine_similar(sen_vector, summarize_vector)
                         for summarize_vector in summarize_vectors]))
                    MMRscore[sen] = mmrscore
            max_mmrscore = sorted(MMRscore.items(), key=operator.itemgetter(1), reverse=True)[0][0]
            summarize_set.append(max_mmrscore)
            summarize_set_str = ''.join(summarize_set)
            # if len(summarize_set_str) > 200:
            #     summarize_set.pop(-1)
            #     break
        res = [(summ, index_solution[summ]) for summ in summarize_set]
        return sorted(res, key=lambda x: x[1])

def baidu_ai_summary(text, max_len):
    appid = '17979964'
    appkey = 'Dzpi3spFHhmWmrNOPfW2Yy4B'
    session_key = '9VXAMP32gmKxwPDRDS0CFkGDKPioWhfN'
    res_summary = ''
    clint = nlp.AipNlp(appid, appkey, session_key)
    res_baidu_summary = clint.newsSummary(content=text, max_summary_len=max_len)
    if 'summary' in res_baidu_summary:
        ref_summary = [res_baidu_summary['summary']]
    return ref_summary


if __name__ == '__main__':
    with open("../static/news.txt", "r", encoding='utf-8') as myfile:
        text = myfile.read().replace('\n', '')
    title_ = '中华人民共和国成立70周年庆祝活动总结会议在京举行 习近平亲切会见庆祝活动筹办工作有关方面代表'
    #
    # # 利用textrank
    # trs = TextRankSummarization()
    # txk_result = trs.get_result_simple(text)
    # print(txk_result)


    # print('*' * 100)
    # print(summarize(text, 2))

    sif = SIFSummarization(text)
    sif_result_, sorted_score_, sentence_list_ = sif.main(flags=0)
    print(sif_result_, end='\n')

    # sif = SIFSummarization(text, title_)
    # sif_result, sorted_score, sentence_list = sif.main(flags=1)
    # print(sif_result, sorted_score, sep='\n')

    # tr4s = TextRank4Sentence()
    # tr4s.analyze(text=text, lower=True, source='no_stop_words')
    # key_sentences = tr4s.get_key_sentences(num=10, sentence_min_len=2)
    # for sentence in key_sentences:
    #     print(sentence['weight'], sentence['sentence'])

    # mmr = MMRSummarization()
    # sen_list = text.strip().split("。")
    # sen_list.remove("")
    # doc_list = [jieba.lcut(i) for i in sen_list] # [[word1, word2, ...], [word1, word2, ...], ...]
    # corpus = [" ".join(i) for i in doc_list] # ['word1 word2 word3 ...', 'word1 word2 word3 ...', ...]
    # for i in mmr.MMR(doc_list,corpus):
    #     print(i)
    # print('**' * 100)

    # mmr = MMRSummarization1()
    # for i in mmr.MMR(sentence=text):
    #     print(i)
    #
    max_len = len(text) // 10
    ref_summary = baidu_ai_summary(text, max_len)

    mmr = MMRSummarization()
    res_summary = ''
    for sen, i in mmr.MMR(sentence=text, max_len=max_len):
        res_summary += ''.join(sen.split(' '))
        res_summary += '。'
    precision_score, recall_score = rouge_n(2, ref_summarys=ref_summary, candidate_summary=res_summary)
    print('MMR算法结果及得分。')
    print(precision_score)
    print(recall_score)
    print(res_summary)
    print(ref_summary)
    print('**' * 100)
    print('SIF算法结果及得分。')
    precision_score, recall_score = rouge_n(2, ref_summarys=ref_summary, candidate_summary=sif_result_)
    print('精准度：',precision_score)
    print('召回率：', recall_score)
    print(sif_result_)
    print(ref_summary)

