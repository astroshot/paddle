# coding=utf-8
"""
"""

from __future__ import print_function
import tarfile
import paddle as paddle
import paddle.fluid as fluid
import six
import numpy as np
import math
import collections
import requests
import hashlib
import os
import errno
import shutil
import importlib
import paddle.dataset
import six.moves.cPickle as pickle
import glob
import sys


class DataType(object):
    NGRAM = 1
    SEQ = 2


def word_count(f, word_freq=None):
    if word_freq is None:
        word_freq = collections.defaultdict(int)

    for l in f:
        for w in l.strip().split():
            word_freq[w] += 1
        word_freq['<s>'] += 1
        word_freq['<e>'] += 1

    return word_freq


def reader_creator(filename, word_idx, n, data_type):
    def reader():
        with tarfile.open(paddle.dataset.common.download(paddle.dataset.imikolov.URL, 'imikolov',
                                                         paddle.dataset.imikolov.MD5)) as tf:
            f = tf.extractfile(filename)

            UNK = word_idx['<unk>']
            for l in f:
                if DataType.NGRAM == data_type:
                    assert n > -1, "invalid length"
                    l = ['<s>'] + l.strip().split() + ['<e>']
                    if (len(l) > n):
                        l = [word_idx.get(w, UNK) for w in l]
                        for i in range(n, len(l) + 1):
                            yield tuple(l[i - n:i])
                elif DataType.SEQ == data_type:
                    l = l.strip().split()
                    l = [word_idx.get(w, UNK) for w in l]
                    src_seq = [word_idx['<s>']] + l
                    trg_seq = l + [word_idx['<s>']]
                    if n > 0 and len(src_seq) > n:
                        continue
                    yield src_seq, trg_seq
                else:
                    assert False, 'unknown data type'

    return reader


def train(word_idx, n, data_type=DataType.NGRAM):
    return reader_creator('./simple-examples/data/ptb.train.txt', word_idx, n, data_type)


def test(word_idx, n, data_type=DataType.NGRAM):
    return reader_creator('./simple-examples/data/ptb.valid.txt', word_idx, n, data_type)


def build_dict(min_word_freq=50):
    train_filename = './simple-examples/data/ptb.train.txt'
    test_filename = './simple-examples/data/ptb.valid.txt'
    with tarfile.open('./data/data11887/simple-examples.tgz') as tf:
        trainf = tf.extractfile(train_filename)
        testf = tf.extractfile(test_filename)
        word_freq = word_count(testf, word_count(trainf))
        if '<unk>' in word_freq:
            # remove <unk> for now, since we will set it as last index
            del word_freq['<unk>']

        word_freq = [
            x for x in six.iteritems(word_freq) if x[1] > min_word_freq
        ]

        word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*word_freq_sorted))
        word_idx = dict(list(zip(words, six.moves.range(len(words)))))
        word_idx['<unk>'] = len(words)

    return word_idx


def convert(path):
    N = 5
    word_dict = build_dict()

    paddle.dataset.common.convert(path, train(word_dict, N), 1000, 'imikolov_train')
    paddle.dataset.common.convert(path, test(word_dict, N), 1000, 'imikolov_test')


def inference(words, is_sparse, dict_size, EMBED_SIZE, HIDDEN_SIZE):
    embed1 = fluid.layers.embedding(input=words[0], size=(dict_size, EMBED_SIZE),
                                    dtype='float32', is_sparse=is_sparse, param_attr='shared_w')
    embed2 = fluid.layers.embedding(input=words[0], size=(dict_size, EMBED_SIZE),
                                    dtype='float32', is_sparse=is_sparse, param_attr='shared_w')
    embed3 = fluid.layers.embedding(input=words[0], size=(dict_size, EMBED_SIZE),
                                    dtype='float32', is_sparse=is_sparse, param_attr='shared_w')
    embed4 = fluid.layers.embedding(input=words[0], size=(dict_size, EMBED_SIZE),
                                    dtype='float32', is_sparse=is_sparse, param_attr='shared_w')

    concat_embed = fluid.layers.concat([embed1, embed2, embed3, embed4], axis=1)
    hidden = fluid.layers.fc(input=concat_embed, size=HIDDEN_SIZE, act='sigmoid')
    predict_word = fluid.layers.fc(input=hidden, size=dict_size, act='softmax')
    return predict_word


def train_program(predict_word):
    next_word = paddle.fluid.layers.data('nextw', shape=[1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=predict_word, label=next_word)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost


def optimizer_func():
    return fluid.optimizer.AdagradOptimizer(learning_rate=3e-3,
                                            regularization=fluid.regularizer.L2DecayRegularizer(8e-4))


def train_process(use_cuda, params_dir, is_sparse=True):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    train_reader = paddle.batch(paddle.dataset.imikolov.train(word_dict, N), BATCH_SIZE)
    test_reader = paddle.batch(paddle.dataset.imikolov.test(word_dict, N), BATCH_SIZE)

    first_word = fluid.layers.data('firstw', shape=[1], dtype='int64')
    second_word = fluid.layers.data('secondw', shape=[1], dtype='int64')
    third_word = fluid.layers.data('thirdw', shape=[1], dtype='int64')
    fourth_word = fluid.layers.data('fourthw', shape=[1], dtype='int64')
    next_word = fluid.layers.data('nextw', shape=[1], dtype='int64')

    word_list = [first_word, second_word, third_word, fourth_word, next_word]
    feed_order = ['firstw', 'secondw', 'thirdw', 'fourthw', 'nextw']

    main_program = fluid.default_main_program()
    start_program = fluid.default_startup_program()

    predict_word = inference(word_list, is_sparse, dict_size, EMBED_SIZE, HIDDEN_SIZE)
    avg_cost = train_program(predict_word)
    test_program = main_program.clone(True)

    sgd_optimizer = optimizer_func()
    sgd_optimizer.minimize(avg_cost)

    exe = fluid.Executor(place)

    def train_test(program, reader):
        count = 0
        feed_var_list = [
            program.global_block().var(var_name) for var_name in feed_order
        ]

        feeder_test = fluid.DataFeeder(feed_list=feed_var_list, place=place)
        test_exe = fluid.Executor(place)
        accumulated = len([avg_cost]) * [0]

        for test_data in reader():
            avg_cost_np = test_exe.run(program=program, feed=feeder_test.feed(test_data), fetch_list=[avg_cost])
            accumulated = [x[0] + x[1][0] for x in zip(accumulated, avg_cost_np)]

        return [x / count for x in accumulated]

    def train_loop():
        step = 0
        feed_var_list_loop = [
            main_program.global_block().var(var_name) for var_name in feed_order
        ]

        feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place)
        exe.run(start_program)

        for pass_id in range(PASS_NUM):
            for data in train_reader():
                avg_cost_np = exe.run(main_program, feed=feeder.feed(data), fetch_list=[avg_cost])

                if step % 10 == 0:
                    outs = train_test(test_program, test_reader)
                    print('Step {}, Average Cost {}'.format(step, outs[0]))

                    if outs[0] < 5.8:
                        if params_dir is not None:
                            fluid.io.save_inference_model(params_dir, [
                                'firstw', 'secondw', 'thirdw', 'fourthw'
                            ], [predict_word], exe)
                        return
                step += 1
                if math.isnan(float(avg_cost_np[0])):
                    sys.exit('got NAN loss, training failed')
        raise AssertionError("Cost is too large {0:2.2}".format(avg_cost_np[0]))


def infer(use_cuda, params_dir):
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()




if __name__ == '__main__':
    EMBED_SIZE = 32
    HIDDEN_SIZE = 256
    N = 5
    BATCH_SIZE = 100
    PASS_NUM = 100

    paddle.dataset.common.DATA_HOME = os.path.expanduser('~/cache/paddle/dataset')

    use_cuda = False
    word_dict = build_dict()
    dict_size = len(word_dict)
