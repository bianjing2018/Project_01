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
logger = logging.getLogger()



@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/topic', methods=['GET', 'POST'])
def topic():
    news = request.form.get('news')
    if news:
        news = news.split('。')
        news = [''.join(re.findall(r'[\d|\w]+', new)) for new in news]
        news = '。'.join(news)
        sentence_cos_similar, sentences = compute_sentence_ftidf(news)
        print(sentence_cos_similar)
    return render_template('index.html')