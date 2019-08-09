from flask import Flask
from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)
app.config.from_object('config')
db = SQLAlchemy(app)


class NewsChinese(db.Model):
    __tablename__ = 'news_chinese'

    id = db.Column(db.Integer, primary_key=True)
    author = db.Column(db.String(32))
    source = db.Column(db.String(32))
    content = db.Column(db.String(1000))
    feature = db.Column(db.String(256))
    title = db.Column(db.String(32))
    url = db.Column(db.String(32))


db.create_all()