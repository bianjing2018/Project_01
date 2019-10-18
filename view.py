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
import pandas
import logging, re
from train_model import *
import parse_text
from views import summarization_simple as ss
logger = logging.getLogger()


@app.route('/html.html/')
def html():
    return render_template('html.html')


@app.route('/', methods=['GET', 'POST'])
def index_bak():
    return render_template('index.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    # i = 0
    # datas = NewsChinese.query.all()
    # for data in datas:
    #     try:
    #         content = data.content
    #         source = data.source
    #     except Exception as e:
    #         continue
    # dgd = parse_text.DataGraphDisplay()
    # solution, seen, quanzhong = dgd.get_say_similar_word()
    # for key, value in solution.items():
    #     with open('say_word_similar', 'a') as fw:
    #         fw.write(value[0] + '\n')
    # print(solution)
    return render_template('index.html')


@app.route('/submit/', methods=['GET', 'POST'])
def submit():
    news = request.form.get('news_content')
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
    return render_template('html.html', result=result)


@app.route('/graph', methods=['GET', 'POST'])
def graph():
    draw_graph = parse_text.DataGraphDisplay()
    draw_graph.word_cloud()
    return render_template('index.html')


@app.route('/html2.html/')
def html2():
    return render_template('html2.html')


@app.route('/submit2/', methods=['GET', 'POST'])
def submit2():
    news = request.form.get('news_content')
    news_title = request.form.get('news_title')
    res = ''
    if news:
        res = ss.TextRankSummarization().get_result_simple(news)
        # sif = ss.SIFSummarization(news)
        # sif_result, sorted_score, sentence_list = sif.main(flags=0)
        # res = sif_result
        # print(res)
    if news_title:
        sif = ss.SIFSummarization(news, news_title)
        sif_result, sorted_score, sentence_list = sif.main(flags=1)
        res = sif_result
        # print(res)
    return render_template('html2.html', result=res)