import numpy as pd
import pandas as pd
from tools.base_function import save_to_csv
from tools.fake_news_loader import fit_text
import os, jieba
from sklearn.model_selection import train_test_split
from seq2seqSummary.seq2seq import Seq2SeqSummary

FILEPATH = '/Users/haha/Desktop/NewsSet/'
SAVEPATH = '/Users/haha/Desktop/NewsSet/'
TRAIN_LABEL_FILEPATH = os.path.join(SAVEPATH, 'train_label.csv')
TRAIN_TEXT__FILEPATH = os.path.join(SAVEPATH, 'train_text.csv')
if not os.path.exists(TRAIN_LABEL_FILEPATH):
    save_to_csv(file_path=os.path.join(FILEPATH, 'train_label.txt'), save_path=TRAIN_LABEL_FILEPATH, header='label')
if not os.path.exists(TRAIN_TEXT__FILEPATH):
    save_to_csv(file_path=os.path.join(FILEPATH, 'train_text.txt'), save_path=TRAIN_TEXT__FILEPATH, header='comment')

W2V_EMBEDDING_SIZE = 128
EPOCHS = 100

def main():
    X = pd.read_csv(TRAIN_TEXT__FILEPATH, names=["comment"])
    Y = pd.read_csv(TRAIN_LABEL_FILEPATH, names=['label'])
    X = X['comment'][1:100]
    Y = Y['label'][1:100]
    X = [jieba.lcut(s) for s in X]
    Y = [['<BOS>'] + jieba.lcut(s) + ['<EOS>'] for s in Y]
    config = fit_text(X, Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
    print('训练集大小：', len(x_train))
    print('测试集大小：', len(x_test))

    print('开始 fit ...')
    summarizer = Seq2SeqSummary(config)

    history = summarizer.fit(x_train, y_train, x_test, y_test, epochs=EPOCHS)
    a = 1


















if __name__ == '__main__':
    main()