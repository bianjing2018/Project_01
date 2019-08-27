from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap
app = Flask(__name__)
app.config.from_object('config')
db = SQLAlchemy(app)
bootstrap = Bootstrap(app)
from flask import render_template
from flask import request, redirect, jsonify
from model.models import NewsChinese
import logging, re
from train_model import *
import parse_text
logger = logging.getLogger()



@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/topic', methods=['GET', 'POST'])
def topic():
    news = request.form.get('news')
    if news:
        parse = parse_text.ParseDepend()
        parse.deal_sentence(news)
        # news = parse.deal_sentence(news) # [sent1, sent2, sent3] 处理输入的文本
        # sentence_cos_similar, sentences = compute_sentence_ftidf(news) # [sent1, sent2] 以中文句号分割，计算每两句的相似度。
        # words = []
        # for sentence in sentences:
        #     words.extend(sentence.split(' '))
        # parse = ParseDepend(words=sentences)
        result = parse.get_main() # 句子已存分析
        result = result
        print(result)
    return render_template('index.html', result = result)


@app.route('/graph', methods=['GET', 'POST'])
def graph():
    draw_graph = parse_text.DataGraphDisplay()
    draw_graph.word_cloud()
    return render_template('index.html')