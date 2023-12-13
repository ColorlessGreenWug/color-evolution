import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from matplotlib import pyplot as plt

#map csv columns to colors and colors to columns
colors = ["blue","green","purple","red","yellow"]
colors_to_columns = {
'red':['0','9','10','19','20','29'],
'yellow':['1','2','11','12','21','22'],
'green':['3','4','13','14','23','24'],
'blue':['5','6','15','16','25','26'],
'purple':['7','8','17','18','27','28']
}
columns_to_color = dict()
for c in colors:
    for col in colors_to_columns[c]:
        columns_to_color[col] = c

#read csv
chains = []
for i in range(1,4):
    df = pd.read_csv("data/decoded_chain{}.csv".format(i)).tail(5).iloc[:, 1:]
    chains.append(df)

#perform onehot embedding
def onehot(word):
    mapping = {"p": 0, "b": 1, "t": 2, "d": 3,"k":4, "g":5, "m":6, "n":7, "i":8,"u":9,"e":10,"o":11,"a":12}
    embeddings = []
    for w in word:
        embed = [0] * len(mapping.keys())
        embed[mapping[w]] = 1
        embeddings += embed
    if len(embeddings) < (6*len(mapping.keys())):
        embeddings += [0]*(len(mapping.keys())*2)
    return embeddings

#perform dimension reduction
def dim_reduc(chain,num):
    xs = []
    ys = []
    for column in chain.columns:
        for element in chain[column]:
            ys.append(columns_to_color[column])
            xs.append(onehot(element))
    xs = np.array(xs)
    ys = np.array(ys)

    '''tSNE in 2d'''
    tsne = TSNE(n_components=2, perplexity=12.0, learning_rate=200.0, random_state=42)
    xs = tsne.fit_transform(xs)
    tsne_df = pd.DataFrame(data=xs, columns=['PC1', 'PC2'])
    plt.figure(figsize=(8, 6))
    unique_labels = np.array(colors)
    for label in unique_labels:
        indices = np.where(ys == label)[0]
        plt.scatter(tsne_df.iloc[indices, 0], tsne_df.iloc[indices, 1], label=label,color = label)
    plt.title('tSNE Result of Chain {}'.format(num))
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()  
    plt.grid(True)
    plt.savefig('plots/tSNE Result of Chain {}.png'.format(i))

    '''PCA in 3d'''
    # pca = PCA(n_components=3)  
    # principal_components = pca.fit_transform(xs)
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # unique_labels = np.array(colors)
    # for label in unique_labels:
    #     indices = np.where(ys == label)[0]
    #     ax.scatter(principal_components[indices, 0], 
    #            principal_components[indices, 1], 
    #            principal_components[indices, 2], 
    #            label=label,s=10,color = label)
    # ax.set_xlabel('Principal Component 1')
    # ax.set_ylabel('Principal Component 2')
    # ax.set_zlabel('Principal Component 3')
    # plt.title('3D PCA Results with Labels')
    # plt.legend() 
    # plt.show()

    '''PCA in 2d'''
    # pca = PCA(n_components=2)  
    # xs = pca.fit_transform(xs)
    # pca_df = pd.DataFrame(data=xs, columns=['PC1', 'PC2'])
    # plt.figure(figsize=(8, 6))
    # unique_labels = np.array(colors)
    # for label in unique_labels:
    #     indices = np.where(ys == label)[0]
    #     plt.scatter(pca_df.iloc[indices, 0], pca_df.iloc[indices, 1], label=label,color = label)
    # plt.title('PCA Result with Labels')
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.legend() 
    # plt.grid(True)
    # plt.show()

    '''tSNE in 3d'''
    # tsne = TSNE(n_components=3, perplexity=3.0, learning_rate=200.0, random_state=42)
    # xs = tsne.fit_transform(xs)
    # data = tsne.fit_transform(xs)
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # unique_labels = np.array(colors)
    # for label in unique_labels:
    #     indices = np.where(ys == label)[0]
    #     ax.scatter(data[indices, 0], 
    #            data[indices, 1], 
    #            data[indices, 2], 
    #            label=label,s=10,color = label)
    # ax.set_xlabel('Principal Component 1')
    # ax.set_ylabel('Principal Component 2')
    # ax.set_zlabel('Principal Component 3')
    # plt.title('3D tSNE Results with Labels')
    # plt.legend() 
    # plt.show()

for i in range(3):
    dim_reduc(chains[i],i+1)



