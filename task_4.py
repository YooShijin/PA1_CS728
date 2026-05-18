# imported all the W which were dumped for all the SVD and GloVE
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd

print("Loading CoNLL-2003")

dataset = load_dataset("conll2003", trust_remote_code=True) # trust remote so it automaticaaly downloads otherwise prompt appears

trainSplit = dataset["train"]
testSplit = dataset["test"]

tagNames = trainSplit.features["ner_tags"].feature.names
numLabels = len(tagNames)

print("NER tags:", tagNames)

#createing dataset using pytorch Dataset

# this is the normal case when not using the embeddings
class NerDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, id):
        return self.X[id], self.Y[id]

#this is the case when embeddings 
# uncomment all the Embeddig ones and comment all the normal MLP ones
# class NerDataset(Dataset):
#     def __init__(self, split, word2id):
#         self.X = []
#         self.Y = []

#         for item in split:
#             words = item["tokens"]
#             tags = item["ner_tags"]

#             for w, t in zip(words, tags):
#                 id = word2id[w] if w in word2id else word2id["<UNK>"]
#                 self.X.append(id)
#                 self.Y.append(t)

#     def __len__(self):
#         return len(self.Y)

#     def __getitem__(self, id):
#         return self.X[id], self.Y[id]

# defining the nearral network structure using nn.Module
class NerMLP(nn.Module):
    def __init__(self, inputDim, numLabels):
        super().__init__()
        self.fc1 = nn.Linear(inputDim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, numLabels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Here we are also including the input embedding layer
# class NerMLP(nn.Module):
#     def __init__(self, W, numLabels):
#         super().__init__()

#         vocabSize, embDim = W.shape

#         self.embedding = nn.Embedding(vocabSize, embDim)
#         self.embedding.weight.data.copy_(torch.tensor(W))
#         self.embedding.weight.requires_grad = True

#         self.fc1 = nn.Linear(embDim, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, numLabels)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.embedding(x)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         return x

# run experiment functions so we can run in loop for svd and GloVe for all the dimensions 
# This is for simple mlp 
def runExperiment(embeddingPath, modelName, d):

    print(f"\nStarted training: {modelName} | dim = {d}")

    W = np.load(embeddingPath)

    with open("/kaggle/input/datasets/asmitkr/webinfo/word2id.pkl", "rb") as f:
        word2id = pickle.load(f)

    unkVec = np.mean(W, axis=0)
    word2id["<UNK>"] = len(word2id)
    W = np.vstack([W, unkVec])

    def vectorize(split):
        X = []
        Y = []
        for item in split:
            words = item["tokens"]
            tags = item["ner_tags"]
            for w, t in zip(words, tags):
                idx = word2id[w] if w in word2id else word2id["<UNK>"]
                X.append(W[idx])
                Y.append(t)
        return np.array(X), np.array(Y)

    Xtrain, Ytrain = vectorize(trainSplit)
    Xtest, Ytest = vectorize(testSplit)

    trainLoader = DataLoader(
        NerDataset(Xtrain, Ytrain),
        batch_size=256,
        shuffle=True
    )

    testLoader = DataLoader(
        NerDataset(Xtest, Ytest),
        batch_size=256
    )

    model = NerMLP(d, numLabels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lossFn = nn.CrossEntropyLoss()

    startTime = time.time()
    epochs = 20

    for epoch in range(epochs):
        model.train()
        totalLoss = 0.0

        for xb, yb in trainLoader:
            optimizer.zero_grad()
            out = model(xb)
            loss = lossFn(out, yb)
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()

        avgLoss = totalLoss / len(trainLoader)
        print(f"Epoch {epoch + 1}/{epochs} | Loss = {avgLoss:.4f}")

    latency = time.time() - startTime

    print(f"Finished training: {modelName} | dim = {d} | Time = {latency:.2f}s")

    modelPath = f"{modelName}_d{d}.pt"
    torch.save(model.state_dict(), modelPath)
    print(f"Model saved at: {modelPath}")

    model.eval()
    preds = []
    gold = []

    with torch.no_grad():
        for xb, yb in testLoader:
            out = model(xb)
            yhat = torch.argmax(out, dim=1)
            preds.extend(yhat.numpy())
            gold.extend(yb.numpy())

    acc = accuracy_score(gold, preds)
    f1 = f1_score(gold, preds, average="macro")

    print(f"Evaluation → Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")

    return {
        "model": modelName,
        "dim": d,
        "accuracy": acc,
        "macroF1": f1,
        "latency": latency,
        "modelPath": modelPath
    }


# Below one is for the MLP with the emeddings attatched
# def runExperimentEmbedding(embeddingPath, modelName, d):

#     print(f"\nStarted training: {modelName} | dim = {d}")

#     W = np.load(embeddingPath)

#     with open("/kaggle/input/datasets/asmitkr/webinfo/word2id.pkl", "rb") as f:
#         word2id = pickle.load(f)

#     unkVec = np.mean(W, axis=0)
#     word2id["<UNK>"] = len(word2id)
#     W = np.vstack([W, unkVec])

#     trainData = NerDataset(trainSplit, word2id)
#     testData = NerDataset(testSplit, word2id)

#     trainLoader = DataLoader(trainData, batch_size=256, shuffle=True)
#     testLoader = DataLoader(testData, batch_size=256)

#     model = NerMLP(W, numLabels)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     lossFn = nn.CrossEntropyLoss()

#     startTime = time.time()
#     epochs = 25

#     for epoch in range(epochs):
#         model.train()
#         totalLoss = 0.0

#         for xb, yb in trainLoader:
#             optimizer.zero_grad()
#             out = model(xb)
#             loss = lossFn(out, yb)
#             loss.backward()
#             optimizer.step()
#             totalLoss += loss.item()

#         avgLoss = totalLoss / len(trainLoader)
#         print(f"Epoch {epoch + 1}/{epochs} | Loss = {avgLoss:.4f}")

#     latency = time.time() - startTime
#     print(f"Finished training: {modelName} | dim = {d} | Time = {latency:.2f}s")

#     modelPath = f"{modelName}_d{d}.pt"
#     torch.save(model.state_dict(), modelPath)
#     print(f"Model saved at: {modelPath}")

#     model.eval()
#     preds = []
#     gold = []

#     with torch.no_grad():
#         for xb, yb in testLoader:
#             out = model(xb)
#             yhat = torch.argmax(out, dim=1)
#             preds.extend(yhat.numpy())
#             gold.extend(yb.numpy())

#     acc = accuracy_score(gold, preds)
#     f1 = f1_score(gold, preds, average="macro")

#     print(f"Evaluation → Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")

#     return {
#         "model": modelName,
#         "dim": d,
#         "accuracy": acc,
#         "macroF1": f1,
#         "latency": latency,
#         "modelPath": modelPath
#     }



# Running for all the dimensions in both cases
# SImilarly we can run with runExperimentEmbedding with differnt model names
results = []
dims = [50, 100, 200, 300]

for d in dims:
    res = runExperiment(f"/kaggle/input/datasets/asmitkr/webinfo/glove_W_{d}.npy", "GloVe-MLP-2Layer", d)
    results.append(res)

for d in dims:
    res = runExperiment(f"/kaggle/input/datasets/asmitkr/webinfo/embeddings_{d}.npy", "SVD-MLP-2Layer", d)
    results.append(res)


print("\nFinal Results\n")

for r in results:
    print(
        r["model"],
        "d =", r["dim"],
        "Accuracy =", round(r["accuracy"], 4),
        "Macro-F1 =", round(r["macroF1"], 4),
        "Latency =", round(r["latency"], 2)
    )


# This is just for illustration purpose not needed for assignement
nerLabelMap = {
    0: "O (Outside any named entity)",
    1: "B-PER (Beginning of Person name)",
    2: "I-PER (Inside Person name)",
    3: "B-ORG (Beginning of Organization name)",
    4: "I-ORG (Inside Organization name)",
    5: "B-LOC (Beginning of Location name)",
    6: "I-LOC (Inside Location name)",
    7: "B-MISC (Beginning of Miscellaneous entity)",
    8: "I-MISC (Inside Miscellaneous entity)"
}


# Beolow 2 functions are mentioned in the utility

# predictWordNER_Embedding("London", "glove", 200)
# predictWordNER_Embedding("Google", "svd", 200)
# predictWordNER_Embedding("Apple", "glove", 100)
# predictWordNER_Embedding("Einstein", "svd", 300)
# predictWordNER_Embedding("Apple", "glove", 200)
# predictWordNER_Embedding("Cricket", "svd", 200)
# predictWordNER_Embedding("Rahul", "glove", 100)
# predictWordNER_Embedding("India", "svd", 300)

# predictWordNER_Vector("London", "glove", 200)
# predictWordNER_Vector("Google", "svd", 200)
# predictWordNER_Vector("Apple", "glove", 100)
# predictWordNER_Vector("Einstein", "svd", 300)
# predictWordNER_Vector("Apple", "glove", 200)
# predictWordNER_Vector("Cricket", "svd", 200)
# predictWordNER_Vector("Rahul", "glove", 100)
# predictWordNER_Vector("India", "svd", 300)


# Converting our results in pd Dataframe so to easily plot

dfRes = pd.DataFrame(results)
print(dfRes)


# Plotting our results



plt.figure(figsize=(8,5))

for modelName in dfRes["model"].unique():
    sub = dfRes[dfRes["model"] == modelName]
    plt.plot(
        sub["dim"],
        sub["accuracy"],
        marker="o",
        label=modelName
    )

plt.xlabel("Embedding Dimension")
plt.ylabel("Accuracy")
plt.title("NER Accuracy vs Embedding Dimension")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8,5))

for modelName in dfRes["model"].unique():
    sub = dfRes[dfRes["model"] == modelName]
    plt.plot(
        sub["dim"],
        sub["macroF1"],
        marker="o",
        label=modelName
    )

plt.xlabel("Embedding Dimension")
plt.ylabel("Macro-F1 Score")
plt.title("NER Macro-F1 vs Embedding Dimension")
plt.legend()
plt.grid(True)
plt.show()
