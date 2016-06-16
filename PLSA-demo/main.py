from numpy import *
from pylab import *
from math import *
import re
import time
import codecs


# 读数据集
file = codecs.open('dataset.txt','r','utf-8')
documents = [document.strip() for document in file] 
file.close()

# 文档总数
N = len(documents)

# topic数
K = 10

#==============================================================================
# 统计词语在每篇文档中的出现次数以及总词数
#==============================================================================

# map类型，键是word，值是word在全部documents中出现的总次数
wordCount = {}

# list类型，每个元素是一个map类型对象，键是word，值是word在对应的document中出现的次数
wordCountPerDocument = [];

# 要去除的标点符号的正则表达式
punctuationRegex = '[,.;"?!#-_…()`|“”‘]+'

stopwords = ['a','an', 'after', 'also', 'they', 'man', 'zou', 'can', 'and', 'as', 'up', 'soon', 'be', 'being', 'but', 'by', 'd', 'for', 'from', 'he', 'her', 'his', 'in', 'is', 'more', 'of', 'often', 'the', 'to', 'who', 'with', 'people', 'or', 'it', 'that', 'its', 'are', 'has', 'was', 'on', 'at', 'have', 'into', 'no', 'which']

for d in documents:
    words = d.split()
    wordCountCurrentDoc = {}
    for w in words:
        # 过滤stopwords并小写化
        w = re.sub(punctuationRegex, '', w.lower())
        if len(w)<=1 or re.search('http', w) or re.search('[0-9]', w) or w in stopwords:
            continue
        # 否则统计该词出现次数
        if w in wordCount:
            wordCount[w] += 1
        else:
            wordCount[w] = 1
        if w in wordCountCurrentDoc:
            wordCountCurrentDoc[w] += 1
        else:
            wordCountCurrentDoc[w] = 1
    wordCountPerDocument.append(wordCountCurrentDoc);


#==============================================================================
# 构造词表
#==============================================================================

# map类型，键是word，值是word的编号
dictionary = {}
# map类型，键是word的编号，值是word
dictionaryReverse = {}

index = 0;
for word in wordCount.keys():
    if wordCount[word] > 1:
        dictionary[word] = index;
        dictionaryReverse[index] = word;
        index += 1;

# 词表长度
M = len(dictionary)  

#==============================================================================
# 构造document-word矩阵
#==============================================================================

X = zeros([N, M], int8)

for word in dictionary.keys():
    j = dictionary[word]
    for i in range(0, N):
        if word in wordCountPerDocument[i]:
            X[i, j] = wordCountPerDocument[i][word];


#==============================================================================
# 初始化参数
#==============================================================================

# lamda[i, j] : p(zj|di)
lamda = random([N, K])
for i in range(0, N):
    normalization = sum(lamda[i, :])
    for j in range(0, K):
        lamda[i, j] /= normalization;

# theta[i, j] : p(wj|zi)
theta = random([K, M])
for i in range(0, K):
    normalization = sum(theta[i, :])
    for j in range(0, M):
        theta[i, j] /= normalization;

#==============================================================================
# 定义隐变量的后验概率的矩阵表示
#==============================================================================

# p[i, j, k] : p(zk|di,wj)
p = zeros([N, M, K])

#==============================================================================
# E-Step
#==============================================================================
def EStep():
    for i in range(0, N):
        for j in range(0, M):
            denominator = 0;
            for k in range(0, K):
                p[i, j, k] = theta[k, j] * lamda[i, k];
                denominator += p[i, j, k];
            if denominator == 0:
                for k in range(0, K):
                    p[i, j, k] = 0;
            else:
                for k in range(0, K):
                    p[i, j, k] /= denominator;


#==============================================================================
# M-Step
#==============================================================================
def MStep():
    # 更新参数theta
    for k in range(0, K):
        denominator = 0
        for j in range(0, M):
            theta[k, j] = 0
            for i in range(0, N):
                theta[k, j] += X[i, j] * p[i, j, k]
            denominator += theta[k, j]
        if denominator == 0:
            for j in range(0, M):
                theta[k, j] = 1.0 / M
        else:
            for j in range(0, M):
                theta[k, j] /= denominator
        

    # 更新参数lamda
    for i in range(0, N):
        for k in range(0, K):
            lamda[i, k] = 0
            denominator = 0
            for j in range(0, M):
                lamda[i, k] += X[i, j] * p[i, j, k]
                denominator += X[i, j];
            if denominator == 0:
                lamda[i, k] = 1.0 / K
            else:
                lamda[i, k] /= denominator

def LogLikelihood():
    loglikelihood = 0
    for i in range(0, N):
        for j in range(0, M):
            tmp = 0
            for k in range(0, K):
                tmp += theta[k, j] * lamda[i, k]
            if tmp > 0:
                loglikelihood += X[i, j] * log(tmp)
    print('loglikelihood : ', loglikelihood)

#==============================================================================
# EM algorithm
#==============================================================================
LogLikelihood()
for i in range(0, 20):
    EStep()
    MStep()
    print("[", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), "] After the", i+1, "'s iteration  ", )
    LogLikelihood()


topicwords = []
maxTopicWordsNum = 10
for i in range(0, K):
    topicword = []
    for j in range(0, maxTopicWordsNum):
        maxValue = max(theta[i, :])
        index = -1
        for k in range(0, M):
            if theta[i, k] == maxValue:
                theta[i, k] -= 1
                index = k
                break;
        if theta[i, index] > -1:
            topicword.append(dictionaryReverse[index])
        else:
            break
    topicwords.append(topicword)

for i in range(0, K):
    for j in range(0, M):
        if theta[i, j] < 0:
            theta[i, j] += 1;

