import pickle
import lstm_model
import pandas as pd

with open('model/chars.pkl', 'rb') as inp:
    chars = pickle.load(inp)
word_size = 128
maxlen = 128

model = lstm_model.create_model(maxlen, chars, word_size,True)
model.load_weights('model/model.h5', by_name=True)

import re
import numpy as np

# 转移概率用等概率
zy = {'be': 0.5,
      'bm': 0.5,
      'eb': 0.5,
      'es': 0.5,
      'me': 0.5,
      'mm': 0.5,
      'sb': 0.5,
      'ss': 0.5
      }

zy = {i: np.log(zy[i]) for i in zy.keys()}

# 维特比算法
def viterbi(nodes):
    paths = {'b': nodes[0]['b'], 's': nodes[0]['s']}  # 第一层，只有两个节点
    for layer in range(1, len(nodes)):  # 后面的每一层
        paths_ = paths.copy()  # 先保存上一层的路径
        # node_now 为本层节点， node_last 为上层节点
        paths = {}  # 清空 path
        for node_now in nodes[layer].keys():
            # 对于本层的每个节点，找出最短路径
            sub_paths = {}
            # 上一层的每个节点到本层节点的连接
            for path_last in paths_.keys():
                if path_last[-1] + node_now in zy.keys():  # 若转移概率不为 0
                    sub_paths[path_last + node_now] = paths_[path_last] + nodes[layer][node_now] + zy[
                        path_last[-1] + node_now]
            # 最短路径,即概率最大的那个
            sr_subpaths = pd.Series(sub_paths)
            sr_subpaths = sr_subpaths.sort_values()  # 升序排序
            node_subpath = sr_subpaths.index[-1]  # 最短路径
            node_value = sr_subpaths[-1]  # 最短路径对应的值
            # 把 node_now 的最短路径添加到 paths 中
            paths[node_subpath] = node_value
    # 所有层求完后，找出最后一层中各个节点的路径最短的路径
    sr_paths = pd.Series(paths)
    sr_paths = sr_paths.sort_values()  # 按照升序排序
    return sr_paths.index[-1]  # 返回最短路径（概率值最大的路径）


def simple_cut(s):
    if s:
        r = model.predict(np.array([list(chars[list(s)].fillna(0).astype(int)) + [0] * (maxlen - len(s))]),
                          verbose=False)[0][:len(s)]
        r = np.log(r)
        nodes = [dict(zip(['s', 'b', 'm', 'e'], i[:4])) for i in r]
        t = viterbi(nodes)
        words = []
        for i in range(len(s)):
            if t[i] in ['s', 'b']:
                words.append(s[i])
            else:
                words[-1] += s[i]
        return words
    else:
        return []


not_cuts = re.compile('([\da-zA-Zａ-ｚ]+)|[-＊ｂｃｄｏⅡⅢ（）〈〉∶／‰“”％：。，、？！\.\?,!《》]')


def cut_word(s):
    result = []
    j = 0
    for i in not_cuts.finditer(s):
        result.extend(simple_cut(s[j:i.start()]))
        result.append(s[i.start():i.end()])
        j = i.end()
    result.extend(simple_cut(s[j:]))
    return result


# debug 数据
print(cut_word('你的问题可以更一般地表述为：我们是否能以某种不正当的方式'\
'反对一种在我们看来是严重的不公正，哪怕我们所采取的这种不当方式是相对轻微的、不太重要的。'\
'一般而言，私下看别人的日记是不道德的行为，而利用日记材料就更不能不极其谨慎了。'))

"""

with open('data/test.txt', 'rb') as test_inp:
    texts = test_inp.read().decode('utf-8')
s = texts.split('\r\n')  # 根据换行切分不同的句子

with open('data/ans.txt','w', encoding='utf-8') as test_outp:
    for i in s:
        result = "  ".join(cut_word(i))
        test_outp.writelines(result)
        test_outp.writelines('  \n')
"""