#! /usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np
import os
from text_cnn import TextCNN
from utils import create_batch_data, create_word2vec, files_processing

def train(train_dir,val_dir,labels_file,word2vec_path,batch_size,max_steps,log_step,val_step,snapshot,out_dir):
    '''
    训练...
    :param train_dir: 训练数据目录
    :param val_dir:   val数据目录
    :param labels_file:  labels文件目录
    :param word2vec_path: 词向量模型文件
    :param batch_size: batch size
    :param max_steps:  最大迭代次数
    :param log_step:  log显示间隔
    :param val_step:  测试间隔
    :param snapshot:  保存模型间隔
    :param out_dir:   模型ckpt和summaries输出的目录
    :return:
    '''

    max_sentence_length = 300
    embedding_dim = 128
    filter_sizes = [3, 4, 5, 6]
    num_filters = 200  # Number of filters per filter size
    base_lr=0.001# 学习率
    dropout_keep_prob = 0.5
    l2_reg_lambda = 0.0  # "L2 regularization lambda (default: 0.0)


    allow_soft_placement = True  # 如果你指定的设备不存在，允许TF自动分配设备
    log_device_placement = False  # 是否打印设备分配日志


    print("Loading data...")
    w2vModel = create_word2vec.load_wordVectors(word2vec_path)

    labels_set = files_processing.read_txt(labels_file) # <class 'list'>: ['彩票', '教育', '财经', '娱乐', '股票', '房产', '时尚', '体育', '社会', '星座', '时政', '游戏', '家居', '科技']
    labels_nums = len(labels_set)  # 14

    train_file_list = create_batch_data.get_file_list(file_dir=train_dir, postfix='*.npy') # <class 'list'>: ['/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data0.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data1.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data10.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data11.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data12.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data13.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data14.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data15.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data16.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data17.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data18.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data19.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data2.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data20.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data21.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data22.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data23.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data24.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data25.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data26.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data27.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data28.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data29.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data3.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data30.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data31.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data32.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data33.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data34.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data35.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data36.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data37.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data4.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data5.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data6.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data7.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data8.npy', '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data/train_data9.npy']
    train_batch = create_batch_data.get_data_batch(train_file_list, labels_nums=labels_nums, batch_size=batch_size,
                                                   shuffle=False, one_hot=True)

    val_file_list = create_batch_data.get_file_list(file_dir=val_dir, postfix='*.npy')
    val_batch = create_batch_data.get_data_batch(val_file_list, labels_nums=labels_nums, batch_size=batch_size,
                                                 shuffle=False, one_hot=True)

    print("train data info *****************************")
    train_nums=create_word2vec.info_npy(train_file_list)  # 训练数据总数
    print("val data   info *****************************")
    val_nums = create_word2vec.info_npy(val_file_list) # 验证数量总数
    print("labels_set info *****************************")
    files_processing.info_labels_set(labels_set)  # label - > 数字的映射

    # Training
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement = allow_soft_placement,log_device_placement = log_device_placement) # 配置tf.Session的运行方式 allow_soft_placement：自动选择运行设备，
        sess = tf.Session(config = session_conf)
        with sess.as_default():
            cnn = TextCNN(sequence_length = max_sentence_length,
                          num_classes = labels_nums,
                          embedding_size = embedding_dim,
                          filter_sizes = filter_sizes,
                          num_filters = num_filters,
                          l2_reg_lambda = l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=base_lr)
            # optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                if step % log_step==0:
                    print("training: step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                if writer:
                    writer.add_summary(summaries, step)
                return loss, accuracy

            for i in range(max_steps):
                train_batch_data, train_batch_label = create_batch_data.get_next_batch(train_batch)
                train_batch_data = create_word2vec.indexMat2vector_lookup(w2vModel, train_batch_data)

                train_step(train_batch_data, train_batch_label)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % val_step == 0:
                    val_losses = []
                    val_accs = []
                    # for k in range(int(val_nums/batch_size)):
                    for k in range(100):
                        val_batch_data, val_batch_label = create_batch_data.get_next_batch(val_batch)
                        val_batch_data = create_word2vec.indexMat2vector_lookup(w2vModel, val_batch_data)
                        val_loss, val_acc=dev_step(val_batch_data, val_batch_label, writer=dev_summary_writer)
                        val_losses.append(val_loss)
                        val_accs.append(val_acc)
                    mean_loss = np.array(val_losses, dtype=np.float32).mean()
                    mean_acc = np.array(val_accs, dtype=np.float32).mean()
                    print("--------Evaluation:step {}, loss {:g}, acc {:g}".format(current_step, mean_loss, mean_acc))

                if current_step % snapshot == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


def main():
    # Data preprocess
    if os.path.exists('/root/flag_server'):
        labels_file = '/root/project/TextCNNClassfication_SinaNewsData/THUCNews_models/data/THUCNews_labels.txt'
        word2vec_path = "/root/project/TextCNNClassfication_SinaNewsData/THUCNews_word2Vec/THUCNews_word2Vec_128.model"
        train_dir = '/root/project/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data'
        val_dir = '/root/project/TextCNNClassfication_SinaNewsData/THUCNews_models/data/val_data'
        print('词向量路径是否正确', os.path.exists(word2vec_path))
    else:
        labels_file = '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/THUCNews_labels.txt'
        word2vec_path = "/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_word2Vec/THUCNews_word2Vec_128.model"
        train_dir = '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/train_data'
        val_dir = '/Volumes/Samsung_T5/AI/TextCNNClassfication_SinaNewsData/THUCNews_models/data/val_data'
        print('词向量路径是否正确', os.path.exists(word2vec_path))
    max_steps = 100000  # 迭代次数
    batch_size = 128
    out_dir = "./models"  # 模型ckpt和summaries输出的目录

    train(train_dir=train_dir,
          val_dir=val_dir,
          labels_file=labels_file,
          word2vec_path=word2vec_path,
          batch_size=batch_size,
          max_steps=max_steps,
          log_step=50,
          val_step=500,
          snapshot=1000,
          out_dir=out_dir)


if __name__=="__main__":
    main()


