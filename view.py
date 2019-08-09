from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap
app = Flask(__name__)
app.config.from_object('config')
db = SQLAlchemy(app)
bootstrap = Bootstrap(app)
from flask import render_template
from flask import request, redirect, jsonify
from models import NewsChinese
import logging
logger = logging.getLogger()


@app.route('/index', methods=['GET', 'POST'])
def index():
    res = NewsChinese.query.filter_by(author='夏文辉')
    for r in res:
        logger.info(r.content)

    return render_template('index.html')

