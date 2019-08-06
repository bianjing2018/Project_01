from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap
app = Flask(__name__)
app.config.from_object('config')
db = SQLAlchemy(app)
bootstrap = Bootstrap(app)
from flask import render_template
from flask import request, redirect, jsonify
import hashlib, json, time, os


@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

