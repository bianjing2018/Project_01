from tools.base_function import cosine_similar
import re, jieba
import numpy as np
from gensim.models.word2vec import Word2Vec


class SummarizationSubmodularFunction:
    """summarization  by submodular-function
    """

    def __init__(self, lambd=1e-1):
        self.lambd = lambd
        self.alpha = 1
        self.S = []

    def stopwords(self, sentence):
        """stop word"""
        with open('../static/stopwords', 'r') as fr:
            stopword = fr.read().split('\n')
        result = list()
        for sent in sentence:
            result.append([s for s in jieba.lcut(sent) if s not in stopword])
            # result.append(filter(lambda x: x not in stopword, [s for s in jieba.lcut(sent)]).__next__())
        return result

    def split_sentence(self, sentence, stopwords=None):
        """split"""
        sentence = ''.join(re.findall(r'[^\s]', sentence))
        pattern = re.compile('[。？?!！.]')
        sentence = pattern.sub(' ', sentence).split()
        if stopwords:
            split = self.stopwords(sentence=sentence)
        return sentence

    def load_and_vector_to_sentence(self, sentence, wordpath='/Users/haha/Desktop/Project_01/static/save_file/save_mode2'):
        word2vec = Word2Vec.load(wordpath)

    def l_function(self):
        # 度量覆盖度函数
        def c_i_function(i):
            similar = []
            for j, inx in enumerate(self.S):
                w_ij = cosine_similar(j, i)
                similar.append((j, inx, w_ij))
            return max(similar, key=lambda x: x[2])
        for i in V:
            pass
            min(c_i_function())

    def r_function(self):
        # 度量多样性函数
        pass

    def submodular_function(self):

        value = self.l_function() + self.lambd * self.r_function()


if __name__ == "__main__":
    ssf = SummarizationSubmodularFunction()
    # sent = ssf.split_sentence("""11月27日上午，中共中央总书记、国家主席、中央军委主席习近平出席全军院校长集训开班式并发表重要讲话。今年是落实我军建设发展“十三五”规划、实现国防和军队建设2020年目标任务的攻坚之年。
    #                         今年以来，习近平在不同场合多次就强军兴军发表了一系列重要论述。这些强军话语，振奋军心！""", stopwords=True)
    # print(sent)
    print(ssf.load_and_vector_to_sentence())