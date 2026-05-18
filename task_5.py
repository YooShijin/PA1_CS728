# Run with task2 code only just changed the normal term document corr matrix with the TF-IDF implementatin suing TfidfTransformer

from sklearn.feature_extraction.text import TfidfTransformer

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

# build sparse term-document count matrix
Z_counts = csr_matrix((vals, (rows, cols)), shape=(vocabSize, docCount))

# applying TFIDF weighting
tfidf = TfidfTransformer(norm=None)
Z_tfidf = tfidf.fit_transform(Z_counts)


# For MLP training the same code as task4 was used, we just trained using the simple MLP without the embeddings one.