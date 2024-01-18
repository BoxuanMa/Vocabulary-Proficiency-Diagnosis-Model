import ast
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn import preprocessing
pd.set_option('display.max_columns', None)


# 对样本进行预处理并画图
def plot_embedding(data, label, title):

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)     # 对数据进行归一化处理
    fig = plt.figure()      # 创建图形实例
    ax = plt.subplot(111)       # 创建子图
    # 遍历所有样本
    for i in range(data.shape[0]):
        # 在图中为每个数据点画出标签
        plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10),
                 fontdict={'weight': 'bold', 'size': 7})
    plt.xticks()        # 指定坐标的刻度
    plt.yticks()
    plt.title(title, fontsize=14)
    # 返回值
    return fig


def lingskill():
    train_data = pd.read_csv("item_parameter_lingskillv1.csv")
    train_data = train_data.sort_values(by=['format'])
    train_data = train_data.reset_index()
    print(train_data)
    label = train_data['format']

    train_data['b'] = train_data['b'].apply(ast.literal_eval)

    new_data=pd.DataFrame(train_data['b'].tolist(),columns=['b1','b2','b3','b4'])
    print(new_data)
    return new_data, label


def formatskill():
    train_data = pd.read_csv("item_parameter_formatskill.csv")
    train_data = train_data.sort_values(by=['format'])
    train_data = train_data.reset_index()
    print(train_data)
    label = train_data['format']

    train_data['b'] = train_data['b'].apply(ast.literal_eval)

    new_data = pd.DataFrame(train_data['b'].tolist(), columns=['b1', 'b2', 'b3', 'b4','b5'])
    print(new_data)
    return new_data, label

def wordskill():
    train_data = pd.read_csv("item_parameter_word.csv")
    train_data = train_data.sort_values(by=['format'])
    train_data = train_data.reset_index()
    print(train_data)
    label = train_data['wordId']

    train_data['b'] = train_data['b'].apply(ast.literal_eval)

    new_data = pd.DataFrame(train_data['b'].tolist())
    print(new_data)
    return new_data, label

new_data, label=wordskill()

# new_data, label=lingskill()
# new_data, label=formatskill()

print('Starting compute t-SNE Embedding...')
ts = PCA(n_components=2)

# ts = TSNE(n_components=2, perplexity=113.0,random_state=0)
# t-SNE降维
result = ts.fit_transform(new_data)
    # 调用函数，绘制图像
# fig = plot_embedding(result, label, 't-SNE Embedding of digits')
#     # 显示图像
# plt.show()

scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
result = scaler.fit_transform(result)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot()
# ax.set_title('t-SNE process')
scatter=ax.scatter(result[:,0], result[:,1], c=label, s=10)
# handles, _ = scatter.legend_elements(prop='colors')
# labels=[0,1,2,3,4]
# plt.legend(handles, labels)
plt.show()
print(label)