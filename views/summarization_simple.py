import numpy as np
from collections import Counter
from gensim.models.word2vec import Word2Vec
import networkx
from sklearn.decomposition import PCA
import re
from embadding.deal_text import cut
"""https://www.zhongxiaoping.cn/2019/02/25/SIF%E7%AE%97%E6%B3%95%E8%A7%A3%E6%9E%90/#wu-sif-suan-fa-dai-ma-bu-zou sif算法解析"""
WORD_VECTOR = '../static/save_file/save_mode2'


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
        self.model_word_vector = Word2Vec.load(WORD_VECTOR)
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
            for word in word_list:
                if word in self.model_word_vector:
                    a_value = a / (a + self.get_word_frequency(word))  # smooth inverse frequency, SIF
                    vs = np.add(vs, np.multiply(a_value, self.model_word_vector[word]))  # vs += sif * word_vector
                else:
                    continue

            vs = np.divide(vs, sentence_length)  # weighted average
            sentence_set.append(vs)  # add to our existing re-calculated set of sentences
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
    text = '受到A股被纳入MSCI指数的利好消息刺激A股市场从周三开始再度上演龙马行情周四上午金融股和白马股表现喜人但是尾盘跳水之后仅金融板块仍维系红盘状态分析人士认为' \
          '金融股受益于MSCI纳入A股和低估值而重获资金青睐但是存量资金博弈格局下风格交替的震荡格局料延续流动性改善经济悲观预期修正等有助于支撑板块继而大盘指数逐步向好一' \
          '九再现周四A股市场未能延续周三的上行态势两市成交小幅放量29个中信一级行业中收盘仅银行和非银行金融两个行业指数收红分别上涨180和020从二级行业来看股份制与城商行的涨' \
          '幅最高达到222国有银行上涨082信托及其他上涨064保险板块上涨034证券板块上涨006银行板块25只成分股中共有21只收红其中招商银行涨幅最大上涨666贵阳银行上涨365上海银行' \
          '华夏银行浦发银行和兴业银行的涨幅均超过150非银行金融板块44只成分股中共17只个股上涨其中安信信托中国太保涨幅居前两名分别上涨457和304西水股份华安证券中国人寿和新华' \
          '保险的涨幅也均超过2相对而言券商股多小幅下跌近期对A股市场消息面影响最大的就是MSCI宣布从2018年6月开始将A股纳入MSCI新兴市场指数而其中金融股是占比最大的一个群体国金证券' \
          '李立峰团队指出最新方案中包含的222只成分股中剔除了中等市值非互联互通可交易的股票以及有停牌限制的标的由于纳入了很多大市值AH股A股在MSCIEM中的权重由05上升到了073其中金融板块' \
          '占比最高达到4011泛消费次之占比为2426两个板块涵盖了大部分权重股动态来看由于加入了很多是指占比高的金融公司金融板块的权重增加了近一半其他大部分行业权重都受到了稀释尽管A股被纳入M' \
          'SCI这一利好事件对短期市场情绪有所提振对中长期海外增量资金预期升温但短期内市场量能尚不能有效放大金融股独乐乐情景也就难以持续存量博弈格局下风格交替指数震荡格局难改变光大证券指出利好' \
          '并未引起市场太大的热情两市指数和成交量均较为平淡但市场风格出现了较大变化白马股金融' \
          '股上涨的同时成长股题材股全天低迷这表明市场增量资金依然很少存量资金在不同板块之间腾挪这样的跷跷板格局使得指数难有突破市场中期依旧偏空短期依旧可能维持震荡格局'
    sentence_list = ['受到A股被纳入MSCI指数的利好消息刺激，A股市场从周三开始再度上演龙马行情，周四上午金融股和白马股表现喜人，但是尾盘跳水之后，仅金融板块仍维系红盘状态',
                     '分析人士认为，金融股受益于MSCI纳入A股和低估值而重获资金青睐，但是存量资金博弈格局下，风格交替的震荡格局料延续',
                     '流动性改善、经济悲观预期修正等有助于支撑板块继而大盘指数逐步向好',
                     '29个中信一级行业中，收盘仅银行和非银行金融两个行业指数收红，分别上涨1.80%和0.20%',
                     '其中，招商银行涨幅最大，上涨6.66%，贵阳银行上涨3.65%，上海银行、华夏银行、浦发银行和兴业银行的涨幅均超过1.50%',
                     '非银行金融板块44只成分股中，共17只个股上涨',
                     '其中，安信信托、中国太保涨幅居前两名，分别上涨4.57%和3.04%，西水股份、华安证券、中国人寿和新华保险的涨幅也均超过2%',
                     '而其中，金融股是占比最大的一个群体',
                     '国金证券李立峰团队指出，最新方案中包含的222只成分股中，剔除了中等市值、非互联互通可交易的股票以及有停牌限制的标的，由于纳入了很多大市值AH股，A股在MSCI EM中的权重由0.5%上升到了0.73%',
                     '其中，金融板块占比最高，达到40.11%，泛消费次之，占比为24.26%，两个板块涵盖了大部分权重股',
                     '动态来看，由于加入了很多是指占比高的金融公司，金融板块的权重增加了近一半，其他大部分行业权重都受到了稀释',
                     '存量博弈格局下，风格交替、指数震荡格局难改变',
                     '这表明市场增量资金依然很少，存量资金在不同板块之间腾挪，这样的跷跷板格局使得指数难有突破',
                     '市场中期依旧偏空，短期依旧可能维持震荡格局']
    sentence_list.append(text)
    ss = SIFSummarization(text)
    sentence_vector_list = ss.sentence_to_vec(sentence_list)
    similar = ss.compute_similar_by_cosine(sentence_vector_list)
    print(similar)
    sss = sorted(similar.items(), key=lambda x: x[1], reverse=True)
    sss = sorted(sss[: 3], key=lambda x: x[1], reverse=True)
    summ = ''
    for i, v in sss:
        summ += sentence_list[i]
        summ += ' '


    print(summ)

