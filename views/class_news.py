from gensim.models import Word2Vec, word2vec
import numpy as np
from collections import defaultdict
from model.models import NewsChinese
import re, jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB,MultinomialNB
import pickle
from sklearn.model_selection import train_test_split, cross_val_score


# 数据处理、n-gram、
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
        with open('../static/save_file/pickle_hash_encode', 'wb') as f:
            pickle.dump(self.hash_encode, f)

        with open('../static/save_file/pickle_hash_decode', 'wb') as f:
            pickle.dump(self.hash_decode, f)

    def get_tfidf_vector(self, x_train):
        """词向量tfidf 并保存"""
        cv_m = CountVectorizer()
        cv_x_train = cv_m.fit_transform(x_train)
        tf_m = TfidfTransformer()
        tf_x_train = tf_m.fit_transform(cv_x_train)
        with open('../static/save_file/pickle_cv_m', 'wb') as f:
            pickle.dump(cv_m, f)

        with open('../static/save_file/pickle_cv_x_train', 'wb') as f:
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
        with open('../static/save_file/pickle_cv_m', 'rb') as f:
            cv_m = pickle.load(f)

        with open('../static/save_file/pickle_cv_x_train', 'rb') as f:
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
        with open('../static/save_file/pickle_model', 'wb') as f:
            pickle.dump(model, f)

    def model_predict(self, article):
        """预测model"""
        with open('../static/save_file/pickle_cv_m', 'rb') as f:
            cv_m = pickle.load(f)

        with open('../static/save_filepickle_cv_x_train', 'rb') as f:
            cv_x_train = pickle.load(f)

        with open('../static/save_file/pickle_model', 'rb') as f:
            model = pickle.load(f)

        with open('../static/save_file/pickle_hash_encode', 'rb') as f:
            hash_encode = pickle.load(f)

        with open('../static/save_file/pickle_hash_decode', 'rb') as f:
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
