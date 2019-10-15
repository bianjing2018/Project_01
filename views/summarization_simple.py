import numpy as np
from collections import Counter
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
import networkx
from sklearn.decomposition import PCA
import re
import gensim
from embadding.deal_text import cut
from embadding.base_function import cosine_similar
from sklearn.metrics.pairwise import cosine_similarity
"""https://www.zhongxiaoping.cn/2019/02/25/SIF%E7%AE%97%E6%B3%95%E8%A7%A3%E6%9E%90/#wu-sif-suan-fa-dai-ma-bu-zou sif算法解析"""

WORD_VECTOR = '/Users/bj/Desktop/Documents/Project_01/static/save_file/save_mode2'


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
        summarization = self.get_summarization_simple_with_text_rank(text, constrain=len(text) // 5)
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
    text = """
    土耳其大军挺进叙利亚之后，中东局势瞬间就乱成了一锅粥。
就目前来看，库尔德已经主动倒戈叙利亚政府，双方正准备联手对土耳其发动反攻；担忧难民危机卷土重来的英法德，也多次对安卡拉发出危险信号；一心想要复苏俄罗斯经济的普京，则不声不响间和沙特达成了新的军事合作协议；坐山观虎斗的白宫，心心念念的都是怎样用挑起冲突的方式，去让各大势力为美国利益买单。
对土耳其而言，现在已经陷入进退两难的局面：向前一步是荆棘，面对叙利亚政府军、库尔德武装和随时都可能出手的俄军、伊朗士兵，土耳其实在是心有余而力不足；向后一步是深渊——箭在弦上不得不发。土耳其现在的经济状况不容乐观，如果安卡拉就此向美英法德妥协，那么土耳其民众必然不会轻易作罢，库尔德这个"眼中钉"也会继续威胁到土耳其的稳定。
怎么办？这是埃尔多安不得不直面的难题。

10月13日，默克尔、马克龙、约翰逊分别给埃尔多安打了电话。英法德强调，土耳其应立即停止在叙利亚内的军事行动（请注意是立即，言下之意就是没有讨价还价的余地），否则欧洲多国将中止对土耳其的武器出口和经济帮助。英法德认为，安卡拉突然发动的战争，将导致数十万叙利亚民众流离失所，IS必然也会借机趁火打劫，对欧洲和中东各国而言，这将是难以承受之重。
几乎同时，白宫也对土耳其发出了威胁。美国表示，若安卡拉继续铤而走险，那么美国将让土耳其经济彻底陷入瘫痪，等待土耳其的，只能是吃不了兜着走。
然而，已经打红了眼的土耳其却根本不愿认怂。土耳其在日前表示，试图通过经济制裁、取消武器出口等方式来施压安卡拉的国家，终究只能竹篮打水一场空，土耳其不会接受任何国家的斡旋和调停，安卡拉拒绝与库尔德进行谈判。安卡拉强调，土军将继续在叙利亚内作战，直至将恐怖分子（库尔德武装）彻底清除。


值得一提的是，土耳其还特意对美国放了狠话。10月12日，土耳其外长公然喊话白宫："若安卡拉畏惧被制裁，那么我们就不会打响这场战争。"安卡拉强调，没有谁能摧毁土耳其经济，如果美国真要对土耳其下狠手，那么安卡拉必定以牙还牙。
安卡拉是这么说的，土耳其军队也是这么干的。
据俄新社报道，在土耳其军队的穷追猛打下，库尔德武装可说是损失重大，占据的城镇被接连攻克且不说，伤亡数量也一直在不断增加。此外，土耳其雇佣兵还在12日打死了库尔德赫赫有名的女性领导人哈拉夫，这件事让美法等国感到非常震撼。
报道称，哈拉夫的外交才能非常卓越，曾多次参加包括美法等国代表在内的会议，美国前总统特使麦古尔克表示，土耳其对哈拉夫的"处决"是不可接受的，这已经构成了战争罪。但在土耳其看来，这却是非常成功的军事行动。


无视美英法德威胁、接连向叙利亚政府和库尔德武装放狠话后，已经成为众矢之的的土耳其，居然还在关键时刻干了一件大事，这确实让白宫感到始料未及。但可以肯定的是，按照美国从不甘心哑巴亏的性格，在被土耳其"咣咣咣"的打脸后，白宫一定会做出与之对应的报复举动。
    """
    title_ = '以牙还牙？面对美英法德施压，土耳其终于表态了！态度非常强硬'

    # 利用textrank
    trs = TextRankSummarization()
    txk_result = trs.get_result_simple(text)
    print(txk_result)

    sif = SIFSummarization(text)

    sif_result_, sorted_score_, sentence_list_ = sif.main(flags=0)
    print(sif_result_, end='\n')

    sif = SIFSummarization(text, title_)
    sif_result, sorted_score, sentence_list = sif.main(flags=1)
    print(sif_result, sorted_score, sep='\n')








