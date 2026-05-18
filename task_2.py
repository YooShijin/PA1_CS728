# Ran on kaggle so dataset was present there 
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import pickle
import os
from utils.py import getTopKNeighbors

# initial loading and preprocessing and calculatins are exactly same as task 1

file = "/kaggle/input/cc-news-subset/updated_vocab_document_dict.json"

print("Loading CC-News dataset")

with open(file, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Preparing corpus")

docs = []

for key in data:
    for item in data[key]:
        docs.append(item[1])

df = pd.DataFrame({"text": docs})

print("Total documents:", len(df))

print("Tokenizing text")

def textPreprocess(text):
    text = text.replace("\n", " ")
    return text.split()

df["tokens"] = df["text"].apply(textPreprocess)

print("Using predefined vocabulary")

vocab = list(data.keys())

word2id = {w: i for i, w in enumerate(vocab)}
id2word = {i: w for w, i in word2id.items()}

vocabSize = len(vocab)
docCount = len(df)

print("Vocabulary size:", vocabSize)


#Here we are crating the sparse term corr mat we have used csr_matrix for that

rows = []
cols = []
vals = []

for docId, tokens in enumerate(df["tokens"]):
    freq = {}
    for t in tokens:
        if t in word2id:
            ind = word2id[t]
            freq[ind] = freq.get(ind, 0) + 1

    for ind, count in freq.items():
        rows.append(ind)
        cols.append(docId)
        vals.append(count)

Z = csr_matrix((vals, (rows, cols)), shape=(vocabSize, docCount))

print("Sparse matrix shape:", Z.shape)

saveDir = "svdEmbeddings"
os.makedirs(saveDir, exist_ok=True)

dims = [50, 100, 200, 300]

print("Running Truncated SVD")

#using TruncatedSVD for and then fitting to get our embeddings for all the values of d
#we are also saving the trained W for each d so we can use that in task 4
for d in dims:
    print("Computing embeddings for d =", d)

    svd = TruncatedSVD(n_components=d, random_state=42)
    W = svd.fit_transform(Z)

    np.save(f"{saveDir}/embeddings_{d}.npy", W)

    with open(f"{saveDir}/word2id.pkl", "wb") as f:
        pickle.dump(word2id, f)

    with open(f"{saveDir}/id2word.pkl", "wb") as f:
        pickle.dump(id2word, f)

    print("Saved embeddings for d =", d)

print("All embeddings saved")


W = np.load("svdEmbeddings/embeddings_50.npy")

neighbors = getTopKNeighbors("King", W, word2id, id2word, k=5)
print(neighbors)

W = np.load("svdEmbeddings/embeddings_100.npy")

neighbors = getTopKNeighbors("King", W, word2id, id2word, k=5)
print(neighbors)

W = np.load("svdEmbeddings/embeddings_200.npy")

neighbors = getTopKNeighbors("King", W, word2id, id2word, k=5)
print(neighbors)


W = np.load("svdEmbeddings/embeddings_300.npy")

neighbors = getTopKNeighbors("King", W, word2id, id2word, k=5)
print(neighbors)



W = np.load("svdEmbeddings/embeddings_50.npy")

neighbors = getTopKNeighbors("College", W, word2id, id2word, k=5)
print(neighbors)

W = np.load("svdEmbeddings/embeddings_100.npy")

neighbors = getTopKNeighbors("College", W, word2id, id2word, k=5)
print(neighbors)

W = np.load("svdEmbeddings/embeddings_200.npy")

neighbors = getTopKNeighbors("College", W, word2id, id2word, k=5)
print(neighbors)


W = np.load("svdEmbeddings/embeddings_300.npy")

neighbors = getTopKNeighbors("College", W, word2id, id2word, k=5)
print(neighbors)



W = np.load("svdEmbeddings/embeddings_50.npy")

neighbors = getTopKNeighbors("Cricket", W, word2id, id2word, k=5)
print(neighbors)

W = np.load("svdEmbeddings/embeddings_100.npy")

neighbors = getTopKNeighbors("Cricket", W, word2id, id2word, k=5)
print(neighbors)

W = np.load("svdEmbeddings/embeddings_200.npy")

neighbors = getTopKNeighbors("Cricket", W, word2id, id2word, k=5)
print(neighbors)


W = np.load("svdEmbeddings/embeddings_300.npy")

neighbors = getTopKNeighbors("Cricket", W, word2id, id2word, k=5)
print(neighbors)


W = np.load("svdEmbeddings/embeddings_50.npy")

neighbors = getTopKNeighbors("Apple", W, word2id, id2word, k=5)
print(neighbors)

W = np.load("svdEmbeddings/embeddings_100.npy")

neighbors = getTopKNeighbors("Apple", W, word2id, id2word, k=5)
print(neighbors)

W = np.load("svdEmbeddings/embeddings_200.npy")

neighbors = getTopKNeighbors("Apple", W, word2id, id2word, k=5)
print(neighbors)


W = np.load("svdEmbeddings/embeddings_300.npy")

neighbors = getTopKNeighbors("Apple", W, word2id, id2word, k=5)
print(neighbors)




