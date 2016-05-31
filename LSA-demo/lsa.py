import numpy
np = numpy
from scipy import linalg
from matplotlib import pyplot
plt = pyplot
from pylab import *
from numpy import *
import re

# 文档
documents =[
    "Roronoa Zoro, nicknamed \"Pirate Hunter\" Zoro, is a fictional character in the One Piece franchise created by Eiichiro Oda.",
    "In the story, Zoro is the first to join Monkey D. Luffy after he is saved from being executed at the Marine Base. ",
    "Zoro is an expert swordsman who uses three swords for his Three Sword Style, but is also capable of the one and two-sword styles. ",
    "Zoro seems to be more comfortable and powerful using three swords, but he also uses one sword or two swords against weaker enemies.",
    "In One Piece, Luffy sails from the East Blue to the Grand Line in search of the legendary treasure One Piece to succeed Gol D. Roger as the King of the Pirates. ",
    "Luffy is the captain of the Straw Hat Pirates and along his journey, he recruits new crew members with unique abilities and personalities. ",
    "Luffy often thinks with his stomach and gorges himself to comical levels. ",
    "However, Luffy is not as naive as many people believe him to be, showing more understanding in situations than people often expect. ",
    "Knowing the dangers ahead, Luffy is willing to risk his life to reach his goal to become the King of the Pirates, and protect his crew.",
    "Adopted and raised by Navy seaman turned tangerine farmer Bellemere, Nami and her older sister Nojiko, have to witness their mother being murdered by the infamous Arlong.",
    "Nami, still a child but already an accomplished cartographer who dreams of drawing a complete map of the world, joins the pirates, hoping to eventually buy freedom for her village. ",
    "Growing up as a pirate-hating pirate, drawing maps for Arlong and stealing treasure from other pirates, Nami becomes an excellent burglar, pickpocket and navigator with an exceptional ability to forecast weather.",
    "After Arlong betrays her, and he and his gang are defeated by the Straw Hat Pirates, Nami joins the latter in pursuit of her dream."
]
print(len(documents))
# 停用词
stopwords = ['a','an', 'after', 'also', 'and', 'as', 'be', 'being', 'but', 'by', 'd', 'for', 'from', 'he', 'her', 'his', 'in', 'is', 'more', 'of', 'often', 'the', 'to', 'who', 'with', 'people']
# 要去除的标点符号的正则表达式
punctuation_regex = '[,.;"]+'
# map,key是单词,value是单词出现的文档编号
dictionary = {}

# 当前处理的文档编号
currentDocId = 0

# 依次处理每篇文档
for d in documents:
    words = d.split();
    for w in words:
        # 去标点
        w = re.sub(punct_regex, '', w.lower())
        if w in stopwords:
            continue
        elif w in dictionary:
            dictionary[w].append(currentDocId)
        else:
            dictionary[w] = [currentDocId]
    currentDocId += 1

# 至少出现在两个文档中的单词选为关键词
keywords = [k for k in dictionary.keys() if len(dictionary[k]) > 1]
keywords.sort()
print("keywords:\n", keywords, "\n")

# 生成word-document矩阵
X = np.zeros([len(keywords), currentDocId])
for i, k in enumerate(keywords):
    for d in dictionary[k]:
        X[i,d] += 1


# 奇异值分解
U,sigma,V = linalg.svd(X, full_matrices=True)
   
print("U:\n", U, "\n")
print("SIGMA:\n", sigma, "\n")
print("V:\n", V, "\n")

# 得到降维(降到targetDimension维)后单词与文档的坐标表示
targetDimension = 2
U2 = U[0:, 0:targetDimension]
V2 = V[0:targetDimension, 0:]
sigma2 = np.diag(sigma[0:targetDimension])
print(U2.shape, sigma2.shape, V2.shape)

# 对比原始矩阵与降维结果
X2 = np.dot(np.dot(U2, sigma2), V2);
print("X:\n", X);
print("X2:\n", X2);

# 开始画图
plt.title("LSA")
plt.xlabel(u'x')
plt.ylabel(u'y')

# 绘制单词表示的点
# U2的每一行包含了每个单词的坐标表示(维度是targetDimension)，此处使用前两个维度的坐标画图
for i in range(len(U2)):
    text(U2[i][0], U2[i][1],  keywords[i], fontsize=10)
    print("(", U2[i][0], ",", U2[i][1], ")", keywords[i])
x = U2.T[0]
y = U2.T[1]
plot(x, y, '.')

# 绘制文档表示的点
# V2的每一列包含了每个文档的坐标表示(维度是targetDimension)，此处使用前两个维度的坐标画图
for i in range(len(V2[0])):
    text(V2[0][i], V2[1][i], docText[i], fontsize=10)
    print("(", V2[0][i], ",", V2[1][i], ")", ('D%d' %(i+1)))
x = V[0]
y = V[1]
plot(x, y, 'x')


savefig("D:/1.png", dpi=100)
