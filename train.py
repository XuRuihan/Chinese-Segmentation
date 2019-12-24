# -*- coding:utf-8 -*-

import re
import numpy as np
import pandas as pd
import lstm_model
import theano
theano.config.device = 'gpu'
theano.config.floatX = 'float32'

# 设计模型
word_size = 128
maxlen = 32

# 读入训练数据
with open('data/train.txt', 'rb') as inp:
    texts = inp.read().decode('utf-8')
s = texts.split('\r\n')  # 根据换行切分不同的句子

# 清除句子开头的 “ ，另一方面整理一些错误
def clean(s):
    if '“' not in s:
        return s.replace(' ”', '')
    elif '”' not in s:
        return s.replace('“ ', '')
    elif '‘' not in s:
        return s.replace(' ’', '')
    elif '’' not in s:
        return s.replace('‘ ', '')
    else:
        return s

for i in s:
    i = clean(i)

# 判断输入的内容是否为汉字
def is_Chinese(ch):
    if '\u4e00' <= ch <= '\u9fff':
        return True
    return False

# 给每个字分配标记，采用 s, b, m, e 4-tag
def get_xy(s):
    list1 = []
    list2 = []
    for i in range(len(s)):
        if (is_Chinese(s[i])):
            list1.append(s[i])
            if i == 0 or not is_Chinese(s[i-1]):
                if i != len(s) - 1 and is_Chinese(s[i+1]):
                    list2.append('b')
                else:
                    list2.append('s')
            else:
                if i == len(s) - 1 or not is_Chinese(s[i+1]):
                    list2.append('e')
                else:
                    list2.append('m')
        elif s[i] in '[，。！？、]':
            list1.append(s[i])
            list2.append('s')
    return list1, list2

data = []  # 生成训练样本
label = []

for i in s:
    x = get_xy(i)
    if x:
        data.append(x[0])
        label.append(x[1])

d = pd.DataFrame(index=range(len(data)))
d['data'] = data
d['label'] = label
d = d[d['data'].apply(len) <= maxlen]
d.index = range(len(d))
tag = pd.Series({'s': 0, 'b': 1, 'm': 2, 'e': 3, 'x': 4})

# 统计所有字，对每个字编号
chars = []
for i in data:
    chars.extend(i)

chars = pd.Series(chars).value_counts()
chars[:] = range(1, len(chars) + 1)

# 保存数据
import pickle

with open('model/chars.pkl', 'wb') as outp:
    pickle.dump(chars, outp)
print('** Finished saving the data.')

# 生成适合模型输入的格式
from keras.utils import np_utils

d['x'] = d['data'].apply(lambda x: np.array(list(chars[x]) + [0] * (maxlen - len(x))))


def trans_one(x):
    _ = map(lambda y: np_utils.to_categorical(y, 5), tag[x].values.reshape((-1, 1)))
    _ = list(_)
    _.extend([np.array([[0, 0, 0, 0, 1]])] * (maxlen - len(x)))
    return np.array(_)


d['y'] = d['label'].apply(trans_one)


def train_bilstm():
    print("start train bilstm")
    model = lstm_model.create_model(maxlen, chars, word_size)
    batch_size = 1024
    history = model.fit(np.array(list(d['x'])), np.array(list(d['y'])).reshape((-1, maxlen, 5)), batch_size=batch_size,
                        epochs=30, verbose=2)
    model.save('model/model.h5')

# 开始训练模型
train_bilstm()
